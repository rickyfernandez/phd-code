import phd

cdef class DomainManager:
    def __init__(self):

        self.domain = None
        self.load_balance = None
        self.boundary_condition = None

        # search radius of particles
        self.old_radius = DoubleArray()

        # interior particles 
        self.old_interior_flagged.clear()
        self.new_interior_flagged.clear()

        # exterior particles 
        self.old_exterior_flagged.clear()
        self.new_exterior_flagged.clear()

        if phd._in_parallel:

            self.send_cnts = np.zeros(self.size, dtype=np.int32)
            self.recv_cnts = np.zeros(self.size, dtype=np.int32)

            self.send_disp = np.zeros(self.size, dtype=np.int32)
            self.recv_disp = np.zeros(self.size, dtype=np.int32)

    def initialize(self):
        if not self.domain or\
                not self.load_balance or\
                not self.boundary_condition:
            raise RuntimeError("Not all setters defined in DomainMangaer")

    @check_class(phd.Domain)
    def set_domain(domain):
        '''add boundary condition to list'''
        self.domain = domain

    @check_class(phd.BoundaryConditionBase)
    def set_boundary_condition(boundary_condition):
        '''add boundary condition to list'''
        self.boundary_condition = boundary_condition

    @check_class(phd.LoadBalance)
    def set_load_balance(load_balance):
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

    cdef filter_radius(self, CarrayContainer particles):
        """
        Starting radius for each particle in the simulation.

        Parameters
        ----------
        pc : CarrayContainer
            Particle data
        """
        cdef int i
        cdef double box_size = self.domain.max_length
        cdef DoubleArray r = particles.get_carray("radius")
        cdef double search_radius = self.domain.max_length*\
                self.param_box_fraction

        for i in range(particles.get_number_of_items()):
            if phd._in_parallel:
                raise NotImplemented("filter_radius called")
#                if r.data[i] < 0.:
#                    r.data[i] = self.box_fraction*\
#                            self.load_balance.get_node_width(keys.data[i])
            else:
                if r.data[i] < 0.:
                    r.data[i] = search_radius

    cdef create_ghost_particles(CarrayContainer particles):
        """
        After mesh generation, this method goes through partilce list
        and generates ghost particles and communicates them. This method
        is called by mesh repeatedly until the mesh is complete.
        """
        cdef int i, j, k
        cdef BoundaryParticle p
        cdef np.float64_t *x[3], *v[3]
        cdef int num_flagged_exterior, num_flagged_interior

        # extract particle positions and velocities
        particles.pointer_groups(x, particles.named_groups['position'])
        particles.pointer_groups(v, particles.named_groups['velocity'])

        # number flagged particle from previous call
        num_flagged_interior = self.new_interior_flagged.size()
        num_flagged_exterior = self.new_exterior_flagged.size()

        # resize for copying
        self.old_interior_flagged.resize(num_flagged_interior)
        self.old_exterior_flagged.resize(num_flagged_exterior)

        # copy previously flagged
        for i in range(num_flagged_exterior):
            self.old_exterior_flagged[i] =\
                    self.new_exterior_flagged[i]
        self.new_exterior_flagged.clear()

        for i in range(num_flagged_interior):
            self.old_interior_flagged[i] =\
                    self.new_interior_flagged[i]
        self.new_interior_flagged.clear()

        # create new exterior ghost particles
        for i in range(num_flagged_exterior):

            # extract particle
            j = self.old_exterior_flagged[i]
            for k in range(dim):
                p.x[k] = x[k][j]
                p.v[k] = v[k][j]

            # for processor querying
            p.old_radius = self.old_radius.data[j]
            p.new_radius = radius.data[j]
            p.index = j

            # create ghost if needed
            if self.boundary.create_ghost_particle(p, self):
                self.new_exterior_flagged.push_back(i)

        for j in range(num_flagged_interior):

            # extract particle
            i = self.old_interior_flagged[j]
            for k in range(dim):
                p.x[k] = x[k][i]
                p.v[k] = v[k][i]

            # for processor querying
            p.old_radius = self.old_radius.data[j]
            p.new_radius = radius.data[i]
            p.index = i

            if self.create_interior_ghost_particle(p, self):
                self.new_interior_flagged.push_back(i)

        # copy particles, put in processor order and export
        self.copy_particles(particles, ghost_vec)

    cdef copy_particles(CarrayContainer particles, vector[BoundaryParticle] ghost_vec):
        """
        Copy particles that where flagged for ghost creation and append them into particle
        container.
        """
        if phd._in_parallel:
            raise NotImplemented("copy_particles called!")
            self.copy_particles_parallel(particles, ghost_vec)
        else:
            self.copy_particles_serial(particles, ghost_vec)

    cdef copy_particles_serial(CarrayContainer particles, vector[BoundaryParticle] ghost_vec):
        """
        Copy particles from ghost_particle vector
        """
        cdef CarrayContainer ghosts
        cdef DoubleArray mass

        cdef int i, j
        cdef BoundaryParticle *p
        cdef CarrayContainer ghosts
        cdef LongArray indices = LongArray()
        cdef np.float64_t *x[3], *v[3], *mv[3]
        cdef np.float64_t *xg[3], *vg[3], *mv[3]

        # copy indices
        indices.resize(ghost_vec.size())
        for i in range(ghost_vec.size()):
            p = &ghost_vec[i]
            indices.data[i] = p.index

        ghosts = particles.extract_items(indices)

        # reference to position and velocity
        ghosts.pointer_groups(xg, ghosts.named_groups['position'])
        ghosts.pointer_groups(vg, ghosts.named_groups['velocity'])

        # reference to mass and momentum
        mass = ghosts.get_carray("mass")
        ghosts.pointer_groups(mv, ghosts.named_groups['momentum'])

        # transfer new data to ghost 
        for i in range(self.ghost_vec.size()):
            p = &ghost_vec[i]

            # for ghost to retrieve info later 
            maps.data[i] = p.index
            for j in range(dim):

                # update values
                xg[j][i] = p.x[j]
                vg[j][i] = p.v[j]
                mv[j][i] = mass.data[i]*p.v[j]

        # add new ghost to total ghost container
        particles.append_container(ghosts)

    cdef copy_particles_parallel(CarrayContainer particles, vector[BoundaryParticle] ghost_vec):
        pass
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

    cdef bint ghost_complete(self):
        """
        Return True if no particles have been flagged for ghost
        creation
        """
        if phd._in_parallel:
            raise RuntimeError("not implemented yet")
#            # let all processors know if walk is complete 
#            glb_done[0] = 0
#            loc_done[0] = self.export_interaction.done_processing()
#            self.load_bal.comm.Allreduce([loc_done, MPI.INT], [glb_done, MPI.INT], op=MPI.SUM)
#
#            return (self.new_exterior_flagged.empty() and\
#                    self.new_interior_flagged.empty())
        else:
            return True
