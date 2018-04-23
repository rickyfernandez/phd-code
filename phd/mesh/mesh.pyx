import phd
import logging
import numpy as np
from libc.math cimport sqrt
cimport libc.stdlib as stdlib

from ..mesh.pytess cimport PyTess2d, PyTess3d
from ..utils.particle_tags import ParticleTAGS
from ..containers.containers cimport CarrayContainer
from ..utils.carray cimport DoubleArray, LongArray, IntArray

phdLogger = logging.getLogger("phd")

cdef int REAL = ParticleTAGS.Real

# face fields in 2d 
cdef dict face_vars_2d = {
        "area": "double",
        "pair-i": "long",
        "pair-j": "long",
        "com-x": "double",
        "com-y": "double",
        "velocity-x": "double",
        "velocity-y": "double",
        "normal-x": "double",
        "normal-y": "double",
        }

cdef dict named_group_2d = {
        "com": ["com-x", "com-y"],
        "velocity": ["velocity-x", "velocity-y"],
        "normal": ["normal-x", "normal-y"]
        }

# face fields in 3d 
cdef dict face_vars_3d = dict(face_vars_2d, **{
    "com-z":      "double",
    "normal-z":   "double",
    "velocity-z": "double"
    })

cdef dict named_group_3d = {
        "com": ["com-x", "com-y", "com-z"],
        "velocity": ["velocity-x", "velocity-y", "velocity-z"],
        "normal": ["normal-x", "normal-y", "normal-z"]
        }

# particle fields to register in 2d
cdef dict fields_to_register_2d = {
        "volume": "double",
        "dcom-x": "double",
        "dcom-y": "double",
        "w-x"   : "double",
        "w-y"   : "double"
        }

# particle fields to register in 3d
cdef dict fields_to_register_3d = dict(fields_to_register_2d, **{
    "dcom-z": "double",
    "w-z"   : "double"
    })

cdef class Mesh:
    """Voronoi mesh responsible to build mesh, neighbor information,
    and all geometric quantities.

    Attributes
    ----------
    eta : double
        Regularize parameter.

    max_iterations : int
        The max number of mesh updates in a build. This is
        stop an infinite loop for bad meshes.

    num_neighbors : int
        Initial number of neighbors for each particle. Used
        to allocate storage space.

    regularize : bool
        Add regularization to velocity mesh generators.

    relax_iterations : int
        Number of times to perform lloyd relaxation on startup.

    """
    def __init__(self, bint regularize=True, int relax_iterations = 0,
                 double eta=0.25, int num_neighbors=128,
                 max_iterations = 20, **kwargs):
        # domain manager needs to be set
        self.particle_fields_registered = False

        self.max_iterations = max_iterations
        self.relax_iterations = relax_iterations

        self.eta = eta
        self.regularize = regularize
        self.num_neighbors = num_neighbors

    def register_fields(self, CarrayContainer particles):
        """Register mesh fields into the particle container (i.e.
        volume, center of mass).

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef int dim
        cdef str field, dtype
        cdef str axis, dimension
        cdef int num_particles = particles.get_carray_size()

        dim = len(particles.carray_named_groups["position"])
        dimension = "xyz"[:dim]

        particles.carray_named_groups["w"] = []
        particles.carray_named_groups["dcom"] = []

        for axis in dimension:
            particles.carray_named_groups["w"].append("w-" + axis)
            particles.carray_named_groups["dcom"].append("dcom-" + axis)

        if dim == 2:
            self.face_fields = face_vars_2d
            self.face_field_groups = named_group_2d

            for field, dtype in fields_to_register_2d.iteritems():
                if field not in particles.carrays.keys():
                    particles.register_carray(num_particles, field, dtype)

        elif dim == 3:
            self.face_fields = face_vars_3d
            self.face_field_groups = named_group_3d

            for field, dtype in fields_to_register_3d.iteritems():
                if field not in particles.carrays.keys():
                    particles.register_carray(num_particles, field, dtype)

        self.update_ghost_fields = list(particles.carray_named_groups["dcom"])
        self.update_ghost_fields.append("volume")

        # record fields have been registered
        self.particle_fields_registered = True

    def initialize(self):
        """Setup all connections for computation classes. Should check
        always if particle_fields_registered is True.
        """
        cdef int dim
        cdef nn nearest_neigh = nn()

        if not self.particle_fields_registered:
            raise RuntimeError("ERROR: Fields not registered in particles by Mesh!")

        dim = len(self.face_field_groups["velocity"])
        self.neighbors = nn_vec(self.num_neighbors, nearest_neigh)

        if dim == 2:
            self.tess = PyTess2d()
        elif dim == 3:
            self.tess = PyTess3d()

        self.faces = CarrayContainer(carrays_to_register=self.face_fields)
        self.faces.carray_named_groups = self.face_field_groups

    cpdef tessellate(self, CarrayContainer particles, DomainManager domain_manager):
        """Create voronoi tessellation.

        The method of creating the voronoi tessellation follows the idea of
        Efficient Delaunay Tessellation Through K-D Tree Decomposition
        by Dmitriy Morozov and Tom Peterka. The general idea is create
        a local tessellation. Those particles will have either finite
        or infinite radius. Use this radius (for infinite assign a radius)
        and create boundary particles if it intersects the boundary or export
        if intersects another processors boundary. Add new particles to the
        mesh. If the new radius is smaller then previous radius then the
        particle can not influence anymore otherwise increase the radius
        untill all particles are done. In this first implementation we don't
        have a kd tree but use octtree for searches.

        Parameters
        ---------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        cdef int i
        cdef int fail
        cdef np.float64_t *xp[3], *rp
        cdef DoubleArray r = particles.get_carray("radius")
        cdef int start_new_ghost, stop_new_ghost, num_real_particles

        # remove current ghost particles
        particles.remove_tagged_particles(ParticleTAGS.Ghost)

        num_real_particles = particles.get_carray_size()
        start_new_ghost = stop_new_ghost = particles.get_carray_size()

        # reference position and radius 
        rp = r.get_data_ptr()
        particles.pointer_groups(xp, particles.carray_named_groups["position"])

        # first attempt of mesh, radius updated
        assert(self.tess.build_initial_tess(xp, rp, stop_new_ghost) != -1)

        # every infinite radius set to boundary 
        domain_manager.setup_for_ghost_creation(particles)

        for i in range(self.max_iterations):

            # add ghost particles untill mesh is complete
            start_new_ghost = particles.get_carray_size()
            domain_manager.create_ghost_particles(particles)
            stop_new_ghost = particles.get_carray_size()

            # because of malloc
            rp = r.get_data_ptr()
            particles.pointer_groups(xp, particles.carray_named_groups["position"])

            # add ghost particle to mesh
            if start_new_ghost != stop_new_ghost:

                assert(self.tess.update_initial_tess(xp,
                    start_new_ghost, stop_new_ghost) != -1)

                self.tess.update_radius(xp, rp, domain_manager.flagged_particles)

            # update radius of old flagged particles
            domain_manager.update_search_radius(particles)

            # if all process are done flagging
            if domain_manager.ghost_complete():
                break

        if (i+1) == self.max_iterations:
            raise RuntimeError("Mesh failed to converged!")

        # copy radius for next mesh construction
        domain_manager.store_radius(particles)

        if phd._in_parallel:

            # finally reindex ghost in particles and tessellation
            # for correct neighbors will be extracted and exports
            # will be correct
            domain_manager.reindex_ghost(particles, num_real_particles,
                    particles.get_carray_size())
            self.tess.reindex_ghost(domain_manager.import_ghost_buffer)

    cpdef build_geometry(self, CarrayContainer particles, DomainManager domain_manager):
        """Build the voronoi mesh and then extract mesh information, i.e
        volumes, face information, and neighbors.

        Parameters
        ---------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        # particle information
        cdef DoubleArray p_vol = particles.get_carray("volume")

        # face information
        cdef DoubleArray f_area = self.faces.get_carray("area")
        cdef LongArray f_pair_i = self.faces.get_carray("pair-i")
        cdef LongArray f_pair_j = self.faces.get_carray("pair-j")

        # particle pointers
        cdef np.float64_t *x[3], *dcom[3], *vol

        # face pointers
        cdef np.int32_t *pair_i, *pair_j
        cdef np.float64_t *area, *nx[3], *com[3]

        cdef int num_faces, i, j, fail

        phdLogger.info("Mesh: Starting mesh creation")

        # release memory used in the tessellation
        self.reset_mesh()
        self.tessellate(particles, domain_manager)

        # allocate memory for face information
        num_faces = self.tess.count_number_of_faces()
        assert(num_faces != -1)
        self.faces.resize(num_faces)

        # pointers to particle data 
        particles.pointer_groups(x, particles.carray_named_groups["position"])
        particles.pointer_groups(dcom, particles.carray_named_groups["dcom"])
        vol = p_vol.get_data_ptr()

        # pointers to face data
        self.faces.pointer_groups(nx,  self.faces.carray_named_groups["normal"])
        self.faces.pointer_groups(com, self.faces.carray_named_groups["com"])
        pair_i = f_pair_i.get_data_ptr()
        pair_j = f_pair_j.get_data_ptr()
        area   = f_area.get_data_ptr()

        self.neighbors.resize(particles.get_carray_size())
        for i in range(particles.get_carray_size()):
            self.neighbors[i].resize(0)

        # store particle and face information for the tessellation
        # only real particle information is computed
        fail = self.tess.extract_geometry(x, dcom, vol,
                area, com, nx, <int*>pair_i, <int*>pair_j,
                self.neighbors)
        assert(fail != -1)

        # tmp for now
        self.faces.resize(fail)

        # transfer particle information to ghost particles
        domain_manager.update_ghost_fields(particles, self.update_ghost_fields)

    cpdef reset_mesh(self):
        """Clear out mesh data."""
        self.tess.reset_tess()

    cpdef relax(self, CarrayContainer particles, DomainManager domain_manager):
        """Perform mesh relaxation by moving particles to their center of mass.

        Parameters
        ---------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        cdef np.float64_t *x[3], *dcx[3]
        cdef int i, k, dim, num_real_particles
        cdef IntArray tags = particles.get_carray("tag")

        dim = len(particles.carray_named_groups["position"])
        particles.remove_tagged_particles(ParticleTAGS.Ghost)

        # create ghost, extract geometric values
        self.build_geometry(particles, domain_manager)

        # update real particle positions
        particles.pointer_groups(x,   particles.carray_named_groups["position"])
        particles.pointer_groups(dcx, particles.carray_named_groups["dcom"])
        for i in range(particles.get_carray_size()):
            if tags.data[i] == REAL:
                for k in range(dim):
                    x[k][i] += dcx[k][i]

        # use boundary conditions for particles
        # that leave the domain
        domain_manager.migrate_particles(particles)

    cpdef assign_generator_velocities(self, CarrayContainer particles,
                                      EquationStateBase equation_state):
        """Assigns particle velocities.

        Particle velocities are equal to local fluid velocity plus a
        regularization term. The algorithm is taken from Springel (2009).

        Parameters
        ---------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        eos : EquationStateBase
            Thermodynamic equation of state.

        """
        # particle values
        cdef DoubleArray r = particles.get_carray("density")
        cdef DoubleArray p = particles.get_carray("pressure")
        cdef DoubleArray vol = particles.get_carray("volume")

        # local variables
        cdef int i, k, dim
        cdef double c, d, R
        cdef double eta = self.eta
        cdef np.float64_t *x[3], *v[3], *wx[3], *dcx[3]

        dim = len(particles.carray_named_groups["position"])

        particles.pointer_groups(x,   particles.carray_named_groups["position"])
        particles.pointer_groups(v,   particles.carray_named_groups["velocity"])
        particles.pointer_groups(wx,  particles.carray_named_groups["w"])
        particles.pointer_groups(dcx, particles.carray_named_groups["dcom"])

        for i in range(particles.get_carray_size()):

            for k in range(dim):
                wx[k][i] = v[k][i]

            if self.regularize:

                # sound speed 
                c = equation_state.sound_speed(r.data[i], p.data[i])

                # distance form cell com to particle position
                d = 0.0
                for k in range(dim):
                    d += dcx[k][i]**2
                d = sqrt(d)

                # approximate length of cell
                if dim == 2:
                    R = sqrt(vol.data[i]/np.pi)
                if dim == 3:
                    R = pow(3.0*vol.data[i]/(4.0*np.pi), 1.0/3.0)

                # regularize Eq. 63
                if ((0.9 <= d/(eta*R)) and (d/(eta*R) < 1.1)):
                    for k in range(dim):
                        wx[k][i] += c*dcx[k][i]*(d - 0.9*eta*R)/(d*0.2*eta*R)

                elif (1.1 <= d/(eta*R)):
                    for k in range(dim):
                        wx[k][i] += c*dcx[k][i]/d

    cpdef assign_face_velocities(self, CarrayContainer particles):
        """Assigns velocities to face center of face defined by
        neighboring particles.

        The face velocity is the average of particle velocities
        that define the face plus a residual motion. The algorithm
        is taken from Springel (2009).

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        # face information
        cdef LongArray pair_i = self.faces.get_carray("pair-i")
        cdef LongArray pair_j = self.faces.get_carray("pair-j")

        # local variables
        cdef int i, j, k, n, dim
        cdef double factor, denom
        cdef np.float64_t *x[3], *wx[3], *fv[3], *fij[3]

        dim = len(particles.carray_named_groups["position"])

        # particle information
        particles.pointer_groups(wx, particles.carray_named_groups["w"])
        particles.pointer_groups(x,  particles.carray_named_groups["position"])

        # face information
        self.faces.pointer_groups(fij, self.faces.carray_named_groups["com"])
        self.faces.pointer_groups(fv,  self.faces.carray_named_groups["velocity"])

        # loop over each face in mesh
        for n in range(self.faces.get_carray_size()):

            # particles that define face
            i = pair_i.data[n]
            j = pair_j.data[n]

            # correct face velocity due to residual motion Eq. 32
            factor = denom = 0.0
            for k in range(dim):
                factor += (wx[k][i] - wx[k][j])*(fij[k][n] - 0.5*(x[k][i] + x[k][j]))
                denom  += pow(x[k][j] - x[k][i], 2.0)
            factor /= denom

            # face velocity mean of particle velocities and residual term Eq. 33
            for k in range(dim):
                fv[k][n] = 0.5*(wx[k][i] + wx[k][j]) + factor*(x[k][j] - x[k][i])

    cpdef update_from_fluxes(self, CarrayContainer particles, RiemannBase riemann, double dt):
        """Update conservative variables from fluxes.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        riemann : RiemannBase
            Class that solves the riemann problem.

        dt : double
            Simulation time step.

        """
        # face information
        cdef DoubleArray area = self.faces.get_carray("area")
        cdef LongArray pair_i = self.faces.get_carray("pair-i")
        cdef LongArray pair_j = self.faces.get_carray("pair-j")

        # particle values
        cdef DoubleArray m = particles.get_carray("mass")
        cdef DoubleArray e = particles.get_carray("energy")
        cdef IntArray tags = particles.get_carray("tag")
        cdef LongArray ids = particles.get_carray("ids")

        # flux values
        cdef DoubleArray fm = riemann.fluxes.get_carray("mass")
        cdef DoubleArray fe = riemann.fluxes.get_carray("energy")

        cdef double a
        cdef int i, j, k, n, dim
        cdef np.float64_t *x[3], *wx[3], *mv[3], *fmv[3]

        dim = len(particles.carray_named_groups["position"])

        particles.pointer_groups(mv, particles.carray_named_groups["momentum"])
        riemann.fluxes.pointer_groups(fmv, riemann.fluxes.carray_named_groups["momentum"])

        # update conserved quantities
        for n in range(self.faces.get_carray_size()):

            # particles that make up the face
            i = pair_i.data[n]
            j = pair_j.data[n]

            # area of the face
            a = area.data[n]

            # flux entering cell defined by particle i
            if(tags.data[i] == REAL):
                m.data[i] -= dt*a*fm.data[n]  # mass 
                e.data[i] -= dt*a*fe.data[n]  # energy

                # momentum
                for k in range(dim):
                    mv[k][i] -= dt*a*fmv[k][n]

            # flux leaving cell defined by particle j
            if(tags.data[j] == REAL):
                m.data[j] += dt*a*fm.data[n]  # mass
                e.data[j] += dt*a*fe.data[n]  # energy

                # momentum
                for k in range(dim):
                    mv[k][j] += dt*a*fmv[k][n]

            if m.data[i] <= 0.0 or m.data[j] <=0.0:
                print "mass", m.data[i], m.data[j], ids.data[i], ids.data[j], i, j
                raise RuntimeError("Mass less than zero in flux update")
            if e.data[i] <=0.0 or e.data[j] <=0.0:
                print "energy", e.data[i], e.data[j], ids.data[i], ids.data[j], i, j
                raise RuntimeError("Energy less than zero in flux update")
