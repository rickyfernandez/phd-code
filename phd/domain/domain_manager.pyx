import phd

from libc.math cimport fmin
from cython.operator cimport preincrement as inc

from ..utils.tools import check_class
from ..utils.particle_tags import ParticleTAGS

cdef int Ghost = ParticleTAGS.Ghost

# FIX: ids -> id
cdef dict fields_for_parallel = {
        "key": "longlong",
        "process": "long",
        }

cdef class DomainManager:
    def __init__(self, double param_initial_radius, double param_search_radius_factor=2.0):

        self.param_initial_radius = param_initial_radius
        self.param_search_radius_factor = param_search_radius_factor

        self.domain = None
        self.load_balance = None
        self.boundary_condition = None

        # list of particle to create ghost particles from
        self.flagged_particles.clear()

        if phd._in_parallel:

            # mpi send/receive counts
            self.send_cnts = np.zeros(phd.size, dtype=np.int32)
            self.recv_cnts = np.zeros(phd.size, dtype=np.int32)

            # mpi send/recieve displacements
            self.send_disp = np.zeros(phd.size, dtype=np.int32)
            self.recv_disp = np.zeros(phd.size, dtype=np.int32)

    def register_fields(self, CarrayContainer particles):
        """
        Register mesh fields into the particle container (i.e.
        volume, center of mass)
        """
        cdef str field, dtype
        cdef int num_particles = particles.get_number_of_items()

        if True:
            for field, dtype in fields_for_parallel.iteritems():
                if field not in particles.carray_info.keys():
                    particles.register_property(num_particles, field, dtype)

        particles.register_property(num_particles, 'ids', 'long')
        particles.register_property(num_particles, 'map', 'long')
        particles.register_property(num_particles, 'radius', 'double')
        particles.register_property(num_particles, 'old_radius', 'double')

        # set initial radius for mesh generation
        self.setup_initial_radius(particles)

    def initialize(self):
        if not self.domain or not self.boundary_condition:
                #not self.load_balance or
            raise RuntimeError("Not all setters defined in DomainMangaer")

    #@check_class(phd.DomainLimits)
    def set_domain(self, domain):
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

    cpdef check_for_partition(self, CarrayContainer particles):
        """
        Check if partition needs to called
        """
        pass

    cpdef partition(self, CarrayContainer particles):
        """
        Distribute particles across processors.
        """
        pass

    cpdef setup_initial_radius(self, CarrayContainer particles):
        cdef int i
        cdef DoubleArray r = particles.get_carray("radius")
        cdef DoubleArray rold = particles.get_carray("old_radius")

        for i in range(particles.get_number_of_items()):
            r.data[i] = self.param_initial_radius
            rold.data[i] = self.param_initial_radius

    cpdef setup_for_ghost_creation(self, CarrayContainer particles):
        """
        Go through each particle and flag for ghost creation. For particles
        with infinite radius use domain size (serial) or partition tile
        (parallel) as initial search radius.

        Parameters
        ----------
        pc : CarrayContainer
            Particle data
        """
        cdef int i, k, dim
        cdef FlagParticle *p
        cdef double search_radius
        cdef np.float64_t *x[3], *v[3]
        cdef DoubleArray r = particles.get_carray("radius")
        cdef DoubleArray rold = particles.get_carray("old_radius")

        dim = len(particles.named_groups["velocity"])

        particles.pointer_groups(x, particles.named_groups['position'])
        particles.pointer_groups(v, particles.named_groups['velocity'])

        # set ghost buffer to zero
        self.ghost_vec.clear()

        # fraction of domain size FIX: search radius should be twice old radius
        self.flagged_particles.resize(particles.get_number_of_items(), FlagParticle())

        # there should be no ghost particles
        i = 0
        cdef cpplist[FlagParticle].iterator it = self.flagged_particles.begin()
        while(it != self.flagged_particles.end()):

            # for infinite particles have fraction of domain size or
            # processor tile as initial radius
            if phd._in_parallel:
                raise NotImplemented("parallel function called")
                # FIX: add runtime time method to set initial radius
            else:
                if r.data[i] < 0: # infinite radius
                    # radius from previous step
                    r.data[i] = rold.data[i]

            # populate with particle information
            p = particle_flag_deref(it)
            p.index = i

            # scale search radius from voronoi radius
            p.old_search_radius = 0.  # initial pass 
            #p.search_radius = r.data[i]
            p.search_radius = min(r.data[i], self.param_search_radius_factor*rold.data[i])

            # copy position and velocity
            for k in range(dim):
                p.x[k] = x[k][i]
                p.v[k] = v[k][i]

            # next particle
            inc(it)
            i += 1

    cpdef update_search_radius(self, CarrayContainer particles):
        """
        Go through each flag particle and update its radius. If
        the particle is still infinite double the search radius. If
        the new radius is smaller then the old search radius that
        particle is done.

        Parameters
        ----------
        pc : CarrayContainer
            Particle data
        """
        cdef int i, k
        cdef FlagParticle *p
        cdef double search_radius
        cdef DoubleArray r = particles.get_carray("radius")

        # there should be no ghost particles
        cdef cpplist[FlagParticle].iterator it = self.flagged_particles.begin()
        while(it != self.flagged_particles.end()):

            # populate with particle information
            p = particle_flag_deref(it)
            i = p.index

            if r.data[i] < 0: # infinite radius
                # grow until finite
                p.old_search_radius = p.search_radius
                p.search_radius = self.param_search_radius_factor*p.search_radius
                inc(it) # next particle

            else: # finite radius
                # if updated radius is smaller than
                # then search radius we are done
                if r.data[i] < p.search_radius:
                    it = self.flagged_particles.erase(it)
                else:
                    p.old_search_radius = p.search_radius
                    p.search_radius = self.param_search_radius_factor*r.data[i]
                    inc(it) # next particle

    cpdef create_ghost_particles(self, CarrayContainer particles):
        """
        After mesh generation, this method goes through partilce list
        and generates ghost particles and communicates them. This method
        is called by mesh repeatedly until the mesh is complete.
        """
        cdef int i
        cdef FlagParticle *p

        self.ghost_vec.clear()

        # create particles from flagged particles
        cdef cpplist[FlagParticle].iterator it = self.flagged_particles.begin()
        while it != self.flagged_particles.end():

            # retrieve particle
            p = particle_flag_deref(it)

            # create ghost particles 
            self.boundary_condition.create_ghost_particle(p, self)
            self.create_interior_ghost_particle(p)
            inc(it)  # increment iterator

        # copy particles, put in processor order and export
        self.copy_particles(particles)

    cdef create_interior_ghost_particle(self, FlagParticle* p):
        # not implement yet
        pass

    cdef copy_particles(self, CarrayContainer particles):
        """
        Copy particles that where flagged for ghost creation and append them into particle
        container.
        """
        if phd._in_parallel:
            raise NotImplemented("copy_particles called!")
        else:
            self.copy_particles_serial(particles)

    cdef copy_particles_parallel(self, CarrayContainer particles):
            raise RuntimeError("not implemented yet")

    cdef copy_particles_serial(self, CarrayContainer particles):
        """
        Copy particles from ghost_particle vector
        """
        cdef IntArray tags
        cdef LongArray maps
        cdef DoubleArray mass

        cdef int i, k, dim
        cdef BoundaryParticle *p
        cdef CarrayContainer ghosts
        cdef LongArray indices = LongArray()

        cdef np.float64_t *xg[3], *vg[3], *mv[3]

        dim = len(particles.named_groups['position'])

        if self.ghost_vec.size() == 0:
            return

        # copy indices
        indices.resize(self.ghost_vec.size())
        for i in range(self.ghost_vec.size()):
            p = &self.ghost_vec[i]
            indices.data[i] = p.index

        # copy all particles to make ghost from
        ghosts = particles.extract_items(indices)

        tags = ghosts.get_carray("tag")
        maps = ghosts.get_carray("map")
        mass = ghosts.get_carray("mass")

        ghosts.pointer_groups(mv, particles.named_groups['momentum'])
        ghosts.pointer_groups(xg, particles.named_groups['position'])
        ghosts.pointer_groups(vg, particles.named_groups['velocity'])

        # transfer ghost position and velocity 
        for i in range(self.ghost_vec.size()):
            p = &self.ghost_vec[i]

            maps.data[i] = p.index # reference to image
            tags.data[i] = Ghost   # ghost label

            for k in range(dim):

                # update values
                xg[k][i] = p.x[k]
                vg[k][i] = p.v[k]
                mv[k][i] = mass.data[i]*p.v[k]

        # add new ghost to total ghost container
        particles.append_container(ghosts)

    cpdef bint ghost_complete(self):
        """
        Return True if no particles have been flagged for ghost
        creation
        """
        if phd._in_parallel:
            raise RuntimeError("not implemented yet")
        else:
            return self.flagged_particles.empty()

    cdef values_to_ghost(self, CarrayContainer particles, list fields):
        pass


# ---------------------- add after serial working again -------------------------------
#cdef class DomainManager:
#    cdef filter_radius(self, CarrayContainer particles):
#        for i in range(particles.get_number_of_items()):
#            if phd._in_parallel:
#                raise NotImplemented("filter_radius called")
#                if r.data[i] < 0.:
#                    r.data[i] = self.param_box_fraction*\
#                            self.load_balance.get_node_width(keys.data[i])
#
#    cdef copy_particles_parallel(CarrayContainer particles, vector[BoundaryParticle] ghost_vec):
#        """
#        Copy particles from ghost_particle vector
#        """
#        cdef CarrayContainer ghosts
#        cdef DoubleArray mass
#
#        cdef int i, j
#        cdef BoundaryParticle *p
#        cdef np.float64_t *x[3], *v[3], *mv[3]
#        cdef np.float64_t *xg[3], *vg[3], *mv[3]
#
#        # reset import/export counts
#        for i in range(self.size):
#            self.send_cnts[i] = 0
#            self.recv_cnts[i] = 0
#
#        if ghost_vec.size():
#
#            # sort particles in processor order
#            sort(ghost_vec.begin(), ghost_vec.end(), proc_cmp)
#
#            # copy indices
#            indices.resize(ghost_vec.size())
#            for i in range(ghost_vec.size()):
#                p = &ghost_vec[i]
#
#                indices.data[i] = p.index
#                self.send_cnts[p.proc] += 1
#
#            # transfer updated position and velocity
#            ghosts.pointer_groups(xg, particles.named_groups['position'])
#            ghosts.pointer_groups(vg, particles.named_groups['velocity'])
#            mass = ghosts.get_carray("mass")
#            ghosts.pointer_groups(mv, particles.named_groups['momentum'])
#
#        # transfer new data to ghost 
#        for i in range(ghosts.get_number_of_items()):
#            p = &ghost_vec[i]
#
#            # for ghost to retrieve info later 
#            maps.data[i] = p.index
#            for j in range(dim):
#
#                xg[j][i] = p.x[j]
#                vg[j][i] = p.v[j]
#                mv[j][i] = mass.data[i]*p.v[j]
#
#        # add new ghost to total ghost container
#        self.total_ghost_particles.append_container(exterior_ghost)
#
#        # create ghost particles from flagged particles
#        ghosts = particles.extract_items(indices)
#        self.load_bal.comm.Alltoall([self.send_cnts, MPI.INT],
#                [self.recv_cnts, MPI.INT])
#
#        # how many incoming particles
#        num_import = 0
#        for i in range(self.size):
#            num_import += self.recv_cnts[i]
#
#        # create displacement arrays 
#        self.send_disp[0] = self.recv_disp[0] = 0
#        for i in range(1, self.size):
#            self.send_disp[i] = self.send_cnts[i-1] + self.send_disp[i-1]
#            self.recv_disp[i] = self.recv_cnts[i-1] + self.recv_disp[i-1]
#
#        # send our particles / recieve particles 
#        self.exchange_particles(particles, ghosts,
#                self.send_cnts, self.recv_cnts,
#                0, self.comm,
#                self.pc.named_groups['gravity-walk-export'],
#                self.send_disp, self.recv_disp)
#    cdef bint ghost_complete(self):
#        if phd._in_parallel:
#            raise RuntimeError("not implemented yet")
#            # let all processors know if walk is complete 
#            glb_done[0] = 0
#            loc_done[0] = self.export_interaction.done_processing()
#            self.load_bal.comm.Allreduce([loc_done, MPI.INT], [glb_done, MPI.INT], op=MPI.SUM)
#
#            return (self.new_exterior_flagged.empty() and\
#                    self.new_interior_flagged.empty())
#        else:
#            return True
