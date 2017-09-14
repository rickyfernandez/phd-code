import numpy as np
cimport numpy as np
cimport libc.stdlib as stdlib

from .pytess cimport PyTess2d, PyTess3d
from ..boundary.boundary cimport Boundary
from ..utils.particle_tags import ParticleTAGS
from ..containers.containers cimport CarrayContainer
from ..utils.carray cimport DoubleArray, LongArray, IntArray


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

cdef dict face_vars_3d = face_vars_2d.update({
    "com-z":      "double",
    "normal-z":   "double",
    "velocity-z": "double"
    })

cdef dict fields_to_register_2d = {
        "volume": "double",
        "dcom-x": "double",
        "dcom-y": "double"
        }

cdef dict fields_to_register_3d = fields_to_register_2d.update({
    "dcom-z": "double"
    })

cdef class MeshBase:
    def __init__(self, int param_dim=2, bint param_regularize=True,
            double param_eta=0.25, int param_num_neighbors=128):
        """
        Constructor for Mesh base class.
        """
        # domain manager needs to be set
        self.domain_manager = None
        self.particle_fields_registered = False

        self.param_dim = dim
        self.param_eta = param_eta
        self.param_regularize = param_regularize
        self.param_num_neighbors = param_num_neighbors

    @check_class(DomainManager)
    def set_domain_manager(self, domain_manager):
        '''Set equation of state for gas'''
        self.domain_manager = domain_manager

    def register_fields(self, CarrayContainer particles):
        """
        Register mesh fields into the particle container (i.e.
        volume, center of mass)
        """
        cdef str fields, dtype
        cdef int num_particles = particles.get_number_of_items()

        if self.dim == 2:
            for field, dtype in fields_to_register_2d.iteritems():
                if field not in particles.carray_info.keys():
                    particles.register(num_particles, fields, dtype)

        elif self.dim == 3:
            for field, dtype in fields_to_register_3d.iteritems():
                if field not in particles.carray_info.keys():
                    particles.register(num_particles, fields, dtype)

        # record fields have been registered
        self.particle_fields_registered = True

    def initialize(self):
        """
        """
        cdef nn nearest_neigh = nn()
        self.neighbors = nn_vec(param_num_neigh, nearest_neigh)

        if not self.domain_manager:
            raise RuntimeError("Not all setters defined in Mesh")
        if not self.particle_fields_registered:
            raise RuntimeError("Fields not registered in particles by Mesh")

        if self.dim == 2:
            self.tess = PyTess2d()
            self.faces = CarrayContainer(var_dict=face_vars_2d)
        elif self.dim == 3:
            self.tess = PyTess3d()
            self.faces = CarrayContainer(var_dict=face_vars_3d)
        else:
            raise RuntimeError("Wrong dimension supplied")

    cpdef tessellate(self, CarrayContainer particles):
        """
        Create voronoi mesh by first adding local particles. Then
        using the domain mangager flag particles that are incomplete
        and export them. Continue the process unitil the mesh is
        complete.
        """
        cdef int fail
        cdef np.float64_t *xp[3], *rp
        cdef DoubleArray r = particles.get_carray("radius")

        # remove current ghost particles
        particles.remove_tagged_particles(ParticleTAGS.Ghost)
        num_real_particles = particles.get_number_of_items()

        # reference position and radius 
        rp = r.get_data_ptr()
        particles.pointer_groups(xp, particles.named_groups["position"])

        # flag all particle for ghost creation 
        self.domain_manager.flag_initial_ghost_particles(particles)

        # first attempt of the mesh 
        fail = self.tess.build_initial_tess(xp, rp, num_real_particles, 1.0E33)
        assert(fail != -1)

        first_attempt = True

        while True:

            # add ghost particles untill mesh is complete
            self.domain_manager.create_ghost_particles(particles, first_attempt)
            if self.domain_manager.ghost_not_complete():
                break

            # because of malloc
            rp = r.get_data_ptr()
            particles.pointer_groups(xp, particles.named_groups['position'])

            self.tess.update_initial_tess(xp,
                    num_real_particles,
                    particles.get_number_of_items())

            # update radius of flagged particles 
            self.tess.update_radius(rp, self.domain_manager.new_interior_flagged_particles)
            self.tess.update_radius(rp, self.domain_manager.new_exterior_flagged_particles)

        # remove ghost particles
        num_ghost_particles = particle.get_number_of_items() - num_real_particles
        for i in range(num_ghost_particles):
            self.indices[i] = num_real_particles + i

        # sort particles by processor order
        self.sorted_indices.resize(num_ghost_particles)
        for i in range(num_ghost_particles)
            self.sortd_indices[i].proc  = proc.data[num_real_particles+i]
            self.sortd_indices[i].index = proc.data[num_real_particles+i]
            self.sortd_indices[i].pos = i

        # put ghost in process order for neighbor information
        qsort(<void*> self.sorted_indices, <size_t> self.sorted_indices.size(),
                sizeof(SortedIndex), proc_index_compare)

        for i in range(num_ghost_particles)
            self.indices[i] = self.sorted_indices[i].index

        # reappend ghost particle in ghost order
        ghost_particles = particles.extract_items(self.indices)
        particles.remove_tagged_particles(ParticleTAGS.Ghost)
        particles.append(ghost)

        # map for creating neighbors with ghost
        for i in range(num_ghost_particles)
            self.indices[i] = self.sorted_indices[i].index

    cpdef build_geometry(self, CarrayContainer particles):
        """
        Build the voronoi mesh and then extract mesh information, i.e
        volumes, face information, and neighbors.
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

        cdef int num_faces, i, j, fail, dim = self.dim

        # release memory used in the tessellation
        self.reset_mesh()
        self.tessellate(particles)

        # allocate memory for face information
        num_faces = self.tess.count_number_of_faces()
        self.faces.resize(num_faces)

        # pointers to particle data 
        particles.pointer_groups(x, particles.named_groups['position'])
        particles.pointer_groups(dcom, particles.named_groups['dcom'])
        vol = p_vol.get_data_ptr()

        # pointers to face data
        self.faces.pointer_groups(nx,  self.faces.named_groups['normal'])
        self.faces.pointer_groups(com, self.faces.named_groups['com'])
        pair_i = f_pair_i.get_data_ptr()
        pair_j = f_pair_j.get_data_ptr()
        area   = f_area.get_data_ptr()

        self.neighbors.resize(particles.get_number_of_items())
        for i in range(particles.get_number_of_items()):
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
        self.domain_manager.values_to_ghost(particles, self.fields)

    cpdef reset_mesh(self):
        self.tess.reset_tess()

    cdef assign_generator_velocities(self, CarrayContainer particles):
        """
        Assigns particle velocities. Particle velocities are
        equal to local fluid velocity plus a regularization
        term. The algorithm is taken from Springel (2009).
        """
        # particle values
        cdef DoubleArray vol = particles.get_carray("volume")
        cdef DoubleArray cs  = particles.get_carray('sound-speed')

        # local variables
        cdef double c, d, R
        cdef double eta = self.param_eta
        cdef int i, k, dim = self.param_dim
        cdef np.float64_t *x[3], *v[3], *wx[3], *dcx[3]

        particles.pointer_groups(x,   particles.named_groups['position'])
        particles.pointer_groups(v,   particles.named_groups['velocity'])
        particles.pointer_groups(wx,  particles.named_groups['w'])
        particles.pointer_groups(dcx, particles.named_groups['dcom'])

        for i in range(particles.get_number_of_items()):

            for k in range(dim):
                wx[k][i] = v[k][i]

            if self.param_regularize:

                # sound speed 
                c = cs.data[i]

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

                # regularize - eq. 63
                if ((0.9 <= d/(eta*R)) and (d/(eta*R) < 1.1)):
                    for k in range(dim):
                        wx[k][i] += c*dcx[k][i]*(d - 0.9*eta*R)/(d*0.2*eta*R)

                elif (1.1 <= d/(eta*R)):
                    for k in range(dim):
                        wx[k][i] += c*dcx[k][i]/d

    cdef assign_face_velocities(self, CarrayContainer particles):
        """
        Assigns velocities to the center of mass of the face
        defined by neighboring particles. The face velocity
        is the average of particle velocities that define
        the face plus a residual motion. The algorithm is
        taken from Springel (2009).
        """
        # face information
        cdef LongArray pair_i = self.faces.get_carray("pair-i")
        cdef LongArray pair_j = self.faces.get_carray("pair-j")

        # local variables
        cdef double factor, denom
        cdef int i, j, k, n, dim = self.param_dim
        cdef np.float64_t *x[3], *wx[3], *fv[3], *fij[3]

        particles.pointer_groups(wx, particles.named_groups['w'])
        particles.pointer_groups(x,  particles.named_groups['position'])

        self.faces.pointer_groups(fij, self.faces.named_groups['com'])
        self.faces.pointer_groups(fv,  self.faces.named_groups['velocity'])

        # loop over each face in mesh
        for n in range(self.faces.get_number_of_items()):

            # particles that define face
            i = pair_i.data[n]
            j = pair_j.data[n]

            # correct face velocity due to residual motion - eq. 32
            factor = denom = 0.0
            for k in range(dim):
                factor += (wx[k][i] - wx[k][j])*(fij[k][n] - 0.5*(x[k][i] + x[k][j]))
                denom  += pow(x[k][j] - x[k][i], 2.0)
            factor /= denom

            # the face velocity mean of particle velocities and residual term - eq. 33
            for k in range(dim):
                fv[k][n] = 0.5*(wx[k][i] + wx[k][j]) + factor*(x[k][j] - x[k][i])

    cdef update_from_fluxes(self, CarrayContainer particles, RiemannBase riemann, double dt):
        """Update conserative variables from fluxes"""

        # face information
        cdef DoubleArray area = self.faces.get_carray("area")
        cdef LongArray pair_i = self.faces.get_carray("pair-i")
        cdef LongArray pair_j = self.faces.get_carray("pair-j")

        # particle values
        cdef DoubleArray m  = particles.get_carray("mass")
        cdef DoubleArray e  = particles.get_carray("energy")
        cdef IntArray flags = particles.get_carray("flag")

        # flux values
        cdef DoubleArray fm = riemann.fluxes.get_carray("mass")
        cdef DoubleArray fe = riemann.fluxes.get_carray("energy")

        cdef double a
        cdef int i, j, k, n, dim = self.param_dim
        cdef np.float64_t *x[3], *wx[3], *mv[3], *fmv[3]

        particles.pointer_groups(mv, particles.named_groups['momentum'])
        riemann.fluxes.pointer_groups(fmv, riemann.fluxes.named_groups['momentum'])

        # update conserved quantities
        for n in range(self.faces.get_number_of_items()):

            i = pair_i.data[n]
            j = pair_j.data[n]
            a = area.data[n]

            # flux entering cell defined by particle i
            if(flags.data[i] & REAL):
                m.data[i] -= dt*a*fm.data[n]  # mass 
                e.data[i] -= dt*a*fe.data[n]  # energy

                # momentum
                for k in range(self.dim):
                    mv[k][i] -= dt*a*fmv[k][n]

            # flux leaving cell defined by particle j
            if(flags.data[j] & REAL):
                m.data[j] += dt*a*fm.data[n]  # mass
                e.data[j] += dt*a*fe.data[n]  # energy

                # momentum
                for k in range(dim):
                    mv[k][j] += dt*a*fmv[k][n]
