import phd
import numpy as np

from libc.math cimport fmin
from cython.operator cimport preincrement as inc

from ..load_balance.tree cimport Node

from ..utils.tools import check_class
from ..utils.particle_tags import ParticleTAGS
from ..utils.exchange_particles import exchange_particles


cdef int REAL = ParticleTAGS.Real
cdef int GHOST = ParticleTAGS.Ghost
cdef int EXTERIOR = ParticleTAGS.Exterior
cdef int INTERIOR = ParticleTAGS.Interior


cdef dict fields_for_parallel = {
        "key": "longlong",
        "process": "long",
        }


cdef inline bint boundary_particle_cmp(
        const BoundaryParticle &a, const BoundaryParticle &b) nogil:
    """Sort boundary particles by processor order. This is
    used for exporting particles"""
    return a.proc < b.proc


cdef inline bint ghostid_cmp(
        const GhostID &a, const GhostID &b) nogil:
    """Sort boundary particles by processor order and by export
    order. This is used as a final sort such that imports and
    export have the same order.
    """
    if a.proc < b.proc:
        return True
    elif a.proc > b.proc:
        return False
    else:
        return a.export_num < b.export_num


cdef class DomainManager:
    def __init__(self, double initial_radius,
                 double search_radius_factor=2.0,
                 int load_balance_freq = 5, **kwargs):

        self.initial_radius = initial_radius
        self.load_balance_freq = load_balance_freq
        self.search_radius_factor = search_radius_factor

        self.domain = None
        self.load_balance = None
        self.boundary_condition = None

        self.particle_fields_registered = False

        # list of particle to create ghost particles from
        self.flagged_particles.clear()
        self.num_real_particles = 0
        self.num_export = 0

        if phd._in_parallel:

            self.loc_done = np.zeros(1, dtype=np.int32)
            self.glb_done = np.zeros(1, dtype=np.int32)

            # mpi send/receive counts
            self.send_cnts = np.zeros(phd._size, dtype=np.int32)
            self.recv_cnts = np.zeros(phd._size, dtype=np.int32)

            # mpi send/recieve displacements
            self.send_disp = np.zeros(phd._size, dtype=np.int32)
            self.recv_disp = np.zeros(phd._size, dtype=np.int32)

    def register_fields(self, CarrayContainer particles):
        """Register mesh fields into the particle container (i.e.
        volume, center of mass)

        Parameters
        ----------
        """
        cdef str field, dtype
        cdef int num_particles = particles.get_carray_size()

        if phd._in_parallel:
            for field, dtype in fields_for_parallel.iteritems():
                if field not in particles.carrays.keys():
                    particles.register_carray(num_particles, field, dtype)
        else:
            particles.register_carray(num_particles, "map", "long")

        particles.register_carray(num_particles, "radius", "double")
        particles.register_carray(num_particles, "old_radius", "double")

        # set initial radius for mesh generation
        self.setup_initial_radius(particles)
        self.particle_fields_registered = True

    def initialize(self):
        if not self.particle_fields_registered:
            raise RuntimeError("ERROR: Fields not registered in particles by Mesh!")

        if not self.domain or not self.boundary_condition:
            raise RuntimeError("Not all setters defined in DomainMangaer")

        if phd._in_parallel:
            if not self.load_balance:
                raise RuntimeError("Not all setters defined in DomainMangaer")

    #@check_class(phd.DomainLimits)
    def set_domain_limits(self, domain):
        '''add boundary condition to list'''
        self.domain = domain

    #@check_class(phd.BoundaryConditionBase)
    def set_boundary_condition(self, boundary_condition):
        '''add boundary condition to list'''
        self.boundary_condition = boundary_condition

    #@check_class(phd.LoadBalance)
    def set_load_balance(self, load_balance):
        '''add boundary condition to list'''
        self.load_balance = load_balance

    cpdef check_for_partition(self, CarrayContainer particles, object integrator):
        """
        Check if partition needs to called
        """
        if phd._in_parallel:
            return integrator.iteration % self.load_balance_freq == 0
        else:
            return False

    cpdef partition(self, CarrayContainer particles):
        """
        Distribute particles across processors.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        # for particles that have left the domain apply boundary
        # condition to particle back in the domain
        self.boundary_condition.migrate_particles(particles, self)

        if phd._in_parallel:
            self.load_balance.decomposition(particles)

    cpdef setup_initial_radius(self, CarrayContainer particles):
        """At the start of the simulation assign every particle an
        initial radius used for constructing the mesh. This values
        gets updated after each mesh build.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef int i
        cdef DoubleArray r = particles.get_carray("radius")
        cdef DoubleArray rold = particles.get_carray("old_radius")

        for i in range(particles.get_carray_size()):
            r.data[i] = self.initial_radius
            rold.data[i] = self.initial_radius

    cpdef store_radius(self, CarrayContainer particles):
        """At the start of the simulation assign every particle an
        initial radius used for constructing the mesh. This values
        gets updated after each mesh build.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef int i
        cdef DoubleArray r = particles.get_carray("radius")
        cdef DoubleArray rold = particles.get_carray("old_radius")

        for i in range(particles.get_carray_size()):
            rold.data[i] = r.data[i]

    cpdef setup_for_ghost_creation(self, CarrayContainer particles):
        """Go through each particle and flag for ghost creation. For particles
        with infinite radius use radius from previous time step.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef int i, k, dim
        cdef FlagParticle *p
        cdef double search_radius
        cdef np.float64_t *x[3], *mv[3]
        cdef DoubleArray r = particles.get_carray("radius")
        cdef DoubleArray rold = particles.get_carray("old_radius")

        dim = len(particles.carray_named_groups["position"])

        particles.pointer_groups(x, particles.carray_named_groups["position"])
        particles.pointer_groups(mv, particles.carray_named_groups["momentum"])

        # set ghost buffer to zero
        self.ghost_vec.clear()
        self.num_real_particles = particles.get_carray_size()

        # buffer to keep track of which particles
        # have to exported for ghost updates
        if phd._in_parallel:

            self.num_export = 0
            self.export_ghost_buffer.clear()
            self.import_ghost_buffer.clear()

        # flag all real particles for ghost creation
        # there should be no ghost particles in the particle container
        self.flagged_particles.resize(particles.get_carray_size(), FlagParticle())

        i = 0
        cdef cpplist[FlagParticle].iterator it = self.flagged_particles.begin()
        while(it != self.flagged_particles.end()):

            # populate with particle information
            p = particle_flag_deref(it)
            p.index = i

            # initial values are flags meant to skip
            # calcualtion for first mesh build
            if phd._in_parallel:
                p.old_search_radius = -1
            else:
                p.old_search_radius = 0.

            # scale search radius from voronoi radius
            p.search_radius = self.search_radius_factor*rold.data[i]

            # copy position and momentum, momentum is used because
            # after an update only the momentum is correct
            for k in range(dim):
                p.x[k] = x[k][i]
                p.v[k] = mv[k][i]

            # next particle
            inc(it)
            i += 1

    cpdef update_search_radius(self, CarrayContainer particles):
        """Go through each flag particle and update its radius. If
        the particle is still infinite double the search radius. If
        the new radius is smaller then the old search radius then that
        particle is done.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef int i, k
        cdef FlagParticle *p
        cdef double search_radius
        cdef DoubleArray r = particles.get_carray("radius")

        # there should be no ghost particles
        cdef cpplist[FlagParticle].iterator it = self.flagged_particles.begin()
        while(it != self.flagged_particles.end()):

            # retrieve particle
            p = particle_flag_deref(it)
            i = p.index

            # at this point the radius of each particle has been
            # updated by the mesh

            if r.data[i] < 0: # infinite radius
                # grow until finite
                p.old_search_radius = p.search_radius
                p.search_radius = self.search_radius_factor*p.search_radius
                inc(it) # next particle

            else: # finite radius
                # if updated radius is smaller than
                # then search radius we are done
                if r.data[i] < p.search_radius:
                    it = self.flagged_particles.erase(it)
                else:
                    p.old_search_radius = p.search_radius
                    p.search_radius = self.search_radius_factor*p.search_radius
                    inc(it) # next particle

    cpdef create_ghost_particles(self, CarrayContainer particles):
        """After mesh generation, this method goes through partilce list
        and generates ghost particles and communicates them. This method
        is called by mesh repeatedly until the mesh is complete.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        # clear out for next batch of ghost particles
        self.ghost_vec.clear()
        self.boundary_condition.create_ghost_particle(self.flagged_particles, self)

        # copy particles, put in processor order and export
        if phd._in_parallel:
            # do processor patch ghost particles
            self.create_interior_ghost_particles(particles)
            self.copy_particles_parallel(particles)
        else:
            self.copy_particles_serial(particles)

    cdef create_interior_ghost_particles(self, CarrayContainer particles):
        """Create interior ghost particles to stitch back the solution
        across processors.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef int i, dim
        cdef FlagParticle *p
        cdef LongArray nbrs_pid = LongArray()
        cdef LongArray leaf_pid = self.load_balance.leaf_pid

        dim = len(particles.carray_named_groups["position"])

        # create interior ghost particles from flagged particles
        cdef cpplist[FlagParticle].iterator it = self.flagged_particles.begin()
        while it != self.flagged_particles.end():

            # retrieve particle
            p = particle_flag_deref(it)

            # find all processors encolsed between old_search_radius
            # and search_radius from domain partition
            nbrs_pid.reset()
            self.get_nearest_intersect_process_neighbors(
                    p.x, p.old_search_radius, p.search_radius,
                    phd._rank, nbrs_pid)

            # if processors found put the particle in buffer
            # for ghost creation and export
            if nbrs_pid.length:
                for i in range(nbrs_pid.length):

                    # store particle information for ghost creation
                    self.ghost_vec.push_back(BoundaryParticle(
                        p.x, p.v, p.index, nbrs_pid.data[i], dim))

            inc(it)  # increment iterator

    cdef copy_particles_parallel(self, CarrayContainer particles):
        """Copy particles from ghost_particle vector in parallel run.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef IntArray tags
        cdef IntArray types
        cdef LongLongArray keys

        cdef int i, k, dim, num_import, num_new_ghost

        cdef BoundaryParticle *p
        cdef CarrayContainer ghosts
        cdef LongArray indices = LongArray()

        cdef np.float64_t *xg[3], *mvg[3]

        dim = len(particles.carray_named_groups["position"])

        # reset import/export counts
        for i in range(phd._size):
            self.send_cnts[i] = 0
            self.recv_cnts[i] = 0

        # if ghost created
        num_new_ghost = self.ghost_vec.size()
        if num_new_ghost != 0:

            # sort particles in processor order for export
            sort(self.ghost_vec.begin(), self.ghost_vec.end(),
                    boundary_particle_cmp)

            # copy indices to make ghost particles
            indices.resize(num_new_ghost)
            for i in range(num_new_ghost):

                p = &self.ghost_vec[i]       # retrieve particle
                if p.proc > phd._size or p.proc < 0:
                    raise RuntimeError("Found error in interior ghost")

                indices.data[i] = p.index    # index of particle
                self.send_cnts[p.proc] += 1  # bin processor for export

            # copy particles to make ghost
            ghosts = particles.extract_items(indices)

            tags = ghosts.get_carray("tag")
            types = ghosts.get_carray("type")
            keys = ghosts.get_carray("key")

            # update position and momentum
            ghosts.pointer_groups(mvg, particles.carray_named_groups["momentum"])
            ghosts.pointer_groups(xg, particles.carray_named_groups["position"])

            # transfer new data to ghost 
            for i in range(num_new_ghost):
                p = &self.ghost_vec[i]

               # store export information 
                self.export_ghost_buffer.push_back(GhostID(
                    p.index, p.proc, self.num_export))

                tags.data[i] = GHOST
                types.data[i] = INTERIOR

                # we store export number in the keys data, temporarily
                # for reordering after the mesh is complete
                keys.data[i] = self.num_export
                self.num_export += 1

                for k in range(dim):

                    # update values
                    xg[k][i] = p.x[k]
                    mvg[k][i] = p.v[k] # momentum not velocity

        else:
            ghosts = CarrayContainer(0, particles.carray_dtypes)

        # how many particles are going to each processor
        phd._comm.Alltoall([self.send_cnts, phd.MPI.INT],
                [self.recv_cnts, phd.MPI.INT])

        # how many incoming particles
        num_import = 0
        for i in range(phd._size):
            num_import += self.recv_cnts[i]

        # create displacement arrays 
        self.send_disp[0] = self.recv_disp[0] = 0
        for i in range(1, phd._size):
            self.send_disp[i] = self.send_cnts[i-1] + self.send_disp[i-1]
            self.recv_disp[i] = self.recv_cnts[i-1] + self.recv_disp[i-1]

        # index to start adding new ghost particles
        start_index = particles.get_carray_size()

        # send our particles / recieve particles 
        particles.extend(num_import)
        exchange_particles(particles, ghosts,
                self.send_cnts, self.recv_cnts,
                start_index, phd._comm,
                particles.carrays.keys(),
                self.send_disp, self.recv_disp)

    cdef copy_particles_serial(self, CarrayContainer particles):
        """Copy particles from ghost_particle vector in serial run.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef IntArray tags
        cdef IntArray types
        cdef LongArray maps

        cdef int i, k, dim
        cdef BoundaryParticle *p
        cdef CarrayContainer ghosts
        cdef LongArray indices = LongArray()

        cdef np.float64_t *xg[3], *mvg[3]

        dim = len(particles.carray_named_groups["position"])

        if self.ghost_vec.size() == 0:
            return

        # copy indices
        indices.resize(self.ghost_vec.size())
        for i in range(self.ghost_vec.size()):
            p = &self.ghost_vec[i]
            indices.data[i] = p.index

        # copy all particles to make ghost from
        ghosts = particles.extract_items(indices)

        tags  = ghosts.get_carray("tag")
        types = ghosts.get_carray("type")
        maps  = ghosts.get_carray("map")

        ghosts.pointer_groups(mvg, particles.carray_named_groups["momentum"])
        ghosts.pointer_groups(xg,  particles.carray_named_groups["position"])

        # transfer ghost position and velocity 
        for i in range(self.ghost_vec.size()):
            p = &self.ghost_vec[i]

            maps.data[i]  = p.index  # reference to image
            tags.data[i]  = GHOST    # ghost label
            types.data[i] = INTERIOR

            for k in range(dim):

                # update values
                xg[k][i] = p.x[k]
                mvg[k][i] = p.v[k] # momentum not velocity

        # add new ghost to total ghost container
        particles.append_container(ghosts)

    cpdef bint ghost_complete(self):
        """Return True if their are no more particles flagged for ghost
        creation.

        Particles that have been flagged for ghost creation are stored
        in flagged_particles. When flagged particles have a complete
        voronoi cell they are removed from flagged_particles.

        """
        # we are done when their are no more particles flagged
        if phd._in_parallel:

            self.glb_done[0] = 0
            self.loc_done[0] = self.flagged_particles.size()

            phd._comm.Allreduce(
                    [self.loc_done, phd.MPI.INT],
                    [self.glb_done, phd.MPI.INT],
                    op=phd.MPI.SUM)

            return self.glb_done[0] == 0

        else:
            return self.flagged_particles.empty()

    cpdef move_generators(self, CarrayContainer particles, double dt):
        """Move particles after flux update.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        dt : float
            Time step of the simulation.

        """
        cdef int i, k, dim
        cdef np.float64_t *x[3], *wx[3]
        cdef IntArray tags = particles.get_carray("tag")
        cdef LongArray ids = particles.get_carray("ids")

        dim = len(particles.carray_named_groups["position"])
        particles.pointer_groups(x,  particles.carray_named_groups["position"])
        particles.pointer_groups(wx, particles.carray_named_groups["w"])

        for i in range(particles.get_carray_size()):
            if tags.data[i] == REAL:
                for k in range(dim):
                    x[k][i] += dt*wx[k][i]

    cpdef migrate_particles(self, CarrayContainer particles):
        """For particles that have left the domain or processor patch
        move particles to their appropriate spot.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef Node *node

        cdef IntArray tags
        cdef LongLongArray keys
        cdef LongArray indices = LongArray()

        cdef LongArray leaf_pid
        cdef int incoming_particles
        cdef CarrayContainer export_particles
        cdef int pid, rank = phd._rank, start_index

        cdef double fac
        cdef np.int32_t xh[3]
        cdef np.ndarray corner

        cdef int i, j
        cdef double xp[3], *x[3]
        cdef int dim = self.domain.dim


        particles.remove_tagged_particles(GHOST)
        self.num_real_particles = particles.get_carray_size()

        # for particles that left the domain perform boundary
        # condition on those particles
        self.boundary_condition.migrate_particles(particles, self)

#        if phd._in_parallel:
#
#            # information to map coordinates to hilbert space
#            fac = self.load_balance.fac
#            corner = self.load_balance.corner
#            leaf_pid = self.load_balance.leaf_pid
#
#            tags = particles.get_carray("tag")
#            keys = particles.get_carray("key")
#            particles.pointer_groups(x, particles.carray_named_groups["position"])
#
#            self.num_export = 0
#            self.export_ghost_buffer.clear()
#            # reset import/export counts
#            for i in range(phd._size):
#                self.send_cnts[i] = 0
#                self.recv_cnts[i] = 0
#
#            # got through real particles
#            for i in range(particles.get_carray_size()):
#                if tags.data[i] == REAL:
#
#                    # map to hilbert space
#                    for j in range(dim):
#                        xh[j] = <np.int32_t> ( (x[j][i] - corner[j])*fac )
#
#                    # new hilbert key
#                    keys.data[i] = self.load_balance.hilbert_func(
#                            xh[0], xh[1], xh[2], self.load_balance.order)
#
#                    # find which processor particle lives in
#                    node = self.load_balance.tree.find_leaf(keys.data[i])
#                    pid  = leaf_pid.data[node.array_index]
#
#                    if pid != rank: # flag to export
#                        self.export_ghost_buffer.push_back(GhostID(
#                            i, pid, self.num_export))
#
#                        # bin processor for export
#                        self.send_cnts[pid] += 1
#
#            num_export = self.export_ghost_buffer.size()
#            if num_export != 0:
#
#                # sort our export ghost in processor and export order
#                sort(self.export_ghost_buffer.begin(),
#                        self.export_ghost_buffer.end(), ghostid_cmp)
#
#                indices.resize(num_export)
#                for i in range(num_export):
#                    indices.data[i] = self.export_ghost_buffer[i].index
#
#                export_particles = particles.extract_items(indices)
#                particles.remove_items(indices.get_npy_array())
#
#            else:
#                export_particles = CarrayContainer(0, particles.carray_dtypes)
#
#            # how many particles are going to each processor
#            phd._comm.Alltoall([self.send_cnts, phd.MPI.INT],
#                    [self.recv_cnts, phd.MPI.INT])
#
#            # how many incoming particles
#            incoming_particles = 0
#            for i in range(phd._size):
#                incoming_particles += self.recv_cnts[i]
#
#            # create displacement arrays 
#            self.send_disp[0] = self.recv_disp[0] = 0
#            for i in range(1, phd._size):
#                self.send_disp[i] = self.send_cnts[i-1] + self.send_disp[i-1]
#                self.recv_disp[i] = self.recv_cnts[i-1] + self.recv_disp[i-1]
#
#            # index to start adding new ghost particles
#            start_index = particles.get_carray_size()
#
#            # send our particles / recieve particles 
#            particles.extend(incoming_particles)
#            exchange_particles(particles, export_particles,
#                    self.send_cnts, self.recv_cnts,
#                    start_index, phd._comm,
#                    particles.carrays.keys(),
#                    self.send_disp, self.recv_disp)

    cdef update_ghost_fields(self, CarrayContainer particles, list fields):
        """Transfer ghost fields from their image particle.

        After ghost particles are created their are certain fields that
        cannot be calculated (i.e. volume, center-of-mass ...) and need
        to be upated from their resepective image particle.

        Parameters
        ----------
        particles : CarrayContainer
            Container of particles.

        fields : list
            List of field strings to update

        """
        cdef str field
        cdef CarrayContainer ghosts
        cdef int i, num_ghost_particles
        cdef LongArray indices = LongArray()
        cdef np.ndarray indices_npy, map_indices_npy
        cdef IntArray tags = particles.get_carray("tag")

        if phd._in_parallel:

            num_ghost_particles = self.export_ghost_buffer.size()
            indices.resize(num_ghost_particles)

            # grab indices used to create ghost particles
            for i in range(num_ghost_particles):
                indices.data[i] = self.export_ghost_buffer[i].index

            # export updated fields
            ghosts = particles.extract_items(indices, fields)
            exchange_particles(particles, ghosts,
                    self.send_cnts, self.recv_cnts,
                    self.num_real_particles, phd._comm, fields,
                    self.send_disp, self.recv_disp)

        else:

            # find all ghost that need to be updated
            for i in range(particles.get_carray_size()):
                if tags.data[i] == GHOST:
                    indices.append(i)

            indices_npy = indices.get_npy_array()
            map_indices_npy = particles["map"][indices_npy]

            # update ghost with their image data
            for field in fields:
                particles[field][indices_npy] = particles[field][map_indices_npy]

    cdef update_ghost_gradients(self, CarrayContainer particles, CarrayContainer gradients):
        """Update ghost gradients from their mirror particle.

        After reconstruction only real particles have gradients calculated.
        This call will transfer those calcluated gradients to the respective
        ghost particles with appropriate updates from the boundary condition.

        Parameters
        ----------
        particles : CarrayContainer
            Container of particles.

        gradients : CarrayContainer
            Container of gradients for each primitive field.

        """
        cdef str field
        cdef CarrayContainer grad
        cdef int i, num_ghost_particles
        cdef LongArray indices = LongArray()
        cdef np.ndarray indices_npy, map_indices_npy
        cdef IntArray tags = particles.get_carray("tag")

        if phd._in_parallel:

            num_ghost_particles = self.export_ghost_buffer.size()
            indices.resize(num_ghost_particles)

            # grab indices used to create ghost particles
            for i in range(num_ghost_particles):
                indices.append(self.export_ghost_buffer[i].index)

            grad = particles.extract_items(indices,
                    gradients.carray_named_groups["primitive"])

            exchange_particles(gradients, grad,
                    self.send_cnts, self.recv_cnts, 0, phd._comm,
                    gradients.carray_named_groups["primitive"],
                    self.send_disp, self.recv_disp)

            # modify gradient by boundary condition
            self.boundary_condition.update_gradients(particles,
                    gradients, self)

        else:

            # find all ghost that are outside the domain that
            #need to be updated
            for i in range(particles.get_carray_size()):
                if tags.data[i] == GHOST:
                    indices.append(i)

            # each ghost particle knows the id from which
            # it was created from the map array
            indices_npy = indices.get_npy_array()
            map_indices_npy = particles["map"][indices_npy]

            # update ghost gradient from image particle 
            for field in gradients.carray_named_groups["primitive"]:
                gradients[field][indices_npy] = gradients[field][map_indices_npy]

            # modify gradient by boundary condition
            self.boundary_condition.update_gradients(particles, gradients, self)

    cdef int get_nearest_intersect_process_neighbors(self, double center[3], double old_h,
            double new_h, int rank, LongArray nbrs):
        """
        Gather all processors enclosed in square.

        Parameters
        ----------
        center : double array
            Center of search square in physical space.
        h : double
            Box length of search square in physical space.
        leaf_pid : LongArray
            Array of processors ids, index corresponds to a leaf in global tree.
        rank : int
            Current processor.
        nbrs : LongArray
            Container to hold all the processors ids.
        """
        cdef LongArray old_nbrs = LongArray()
        cdef LongArray new_nbrs = LongArray()

        cdef int i, j, num_new_nbrs, num_old_nbrs
        cdef LongArray leaf_pid = self.load_balance.leaf_pid

        nbrs.reset()

        # find all processors from previous radius
        if old_h > 0:
            self.load_balance.tree.get_nearest_process_neighbors(
                    center, old_h, leaf_pid, rank, old_nbrs)

        # find all processors from new radius
        self.load_balance.tree.get_nearest_process_neighbors(
                center, new_h, leaf_pid, rank, new_nbrs)

        # remove processors that where already flagged
        # in previous pass

        i = j = 0
        num_old_nbrs = old_nbrs.length
        num_new_nbrs = new_nbrs.length

        while(i != num_new_nbrs):
            if j == num_old_nbrs:
                while(i < num_new_nbrs):
                    nbrs.append(new_nbrs.data[i])
                    i += 1
                return nbrs.length

            if new_nbrs.data[i] < old_nbrs.data[j]:
                nbrs.append(new_nbrs.data[i])
                i += 1
            elif new_nbrs.data[i] > old_nbrs.data[j]:
                j += 1
            else:
                i += 1
                j += 1

        return nbrs.length

    cdef reindex_ghost(self, CarrayContainer particles, int num_real_particles,
            int total_num_particles):
        """Since ghost particles are exported in batches in processor order we
        have to sort all particles such that when ghost particle information
        is exported later it arrives exactly in the order which ghost particles
        are in the particle container.
        """
        cdef int i, j
        cdef LongArray procs
        cdef LongLongArray keys
        cdef LongArray indices = LongArray()

        cdef CarrayContainer ghost
        cdef int num_ghost_particles

        num_ghost_particles = total_num_particles - num_real_particles

        # sort our export ghost in processor and export order
        sort(self.export_ghost_buffer.begin(),
                self.export_ghost_buffer.end(), ghostid_cmp)

        procs = particles.get_carray("process")
        keys = particles.get_carray("key")

        j = 0
        # copy ghost information for sort 
        self.import_ghost_buffer.resize(num_ghost_particles)
        for i in range(num_real_particles, total_num_particles):

            self.import_ghost_buffer[j].index = i
            self.import_ghost_buffer[j].proc = procs.data[i]
            self.import_ghost_buffer[j].export_num = keys.data[i]
            j += 1

        # sort our import ghost in processor and export order
        sort(self.import_ghost_buffer.begin(),
                self.import_ghost_buffer.end(), ghostid_cmp)

        # copy particle in correct import order
        indices.resize(num_ghost_particles)
        for i in range(num_ghost_particles):
            indices.data[i] = self.import_ghost_buffer[i].index

        # reappend ghost particles in correct import order 
        ghost = particles.extract_items(indices)
        particles.resize(num_real_particles)
        particles.append_container(ghost)
