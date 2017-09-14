
#cdef int proc_compare(const void *a, const void *b):
#    """
#    Comparison function for sorting PairId struct
#    in processor order.
#    """
#    if( (<PairId*>a).proc < (<PairId*>b).proc ):
#        return -1
#    if( (<PairId*>a).proc > (<PairId*>b).proc ):
#        return 1
#    return 0

cdef class DomainManager:
    def __init__(self):

        self.domain = None
        self.load_balance = None
        self.boundary_condition = None

#        if phd._in_parallel:
#
#            self.send_cnts = np.zeros(phd._size, dtype=np.int32)
#            self.recv_cnts = np.zeros(phd._size, dtype=np.int32)
#
#            self.send_disp = np.zeros(phd._size, dtype=np.int32)
#            self.recv_disp = np.zeros(phd._size, dtype=np.int32)
#
#            # particle id and send processors buffers
#            self.indices = LongArray()

    @check_class(phd.Domain)
    def set_domain(domain):
        '''add boundary condition to list'''
        self.domain = domain

    @check_class(BoundaryConditionBase)
    def set_boundary_condition(boundary_condition):
        '''add boundary condition to list'''
        self.boundary_condition = boundary_condition

    cpdef partition(self, CarrayContainer particles):
        """
        Distribute particles across processors.
        """
        pass

    cdef set_radius(self, CarrayContainer particles, int num_real_particles):
        """
        Filter particle radius. This is done because in the initial mesh creation
        some particles will have infinite radius.

        Parameters
        ----------
        pc : CarrayContainer
            Particle data
        num_real_particles : int
            Number of real particles in the container
        """
        cdef int i
        cdef DoubleArray r = particles.get_carray("radius")
        cdef double box_size = self.domain.max_length
        cdef double fac = self.scale_factor

        for i in range(num_real_particles):
            r.data[i] = min(fac*box_size, r.data[i])

    cdef create_ghost_particles(self, CarrayContainer particles):
        """
        Main routine in creating ghost particles. This routine appends ghost
        particles in particle container. Particle container should not have
        any prior ghost before calling this function.

        Parameters
        ----------
        particles : CarrayContainer
            Particle data
        """
#        if phd._in_parallel:
#            self.create_ghost_particles_parallel(particles)
#        else:
        self.create_ghost_particles_serial(particles)

    cdef int create_ghost_particles_serial(self, CarrayContainer particles):
        """
        Main routine in creating ghost particles. This routine appends ghost
        particles in particle container. Particle container should not have
        any prior ghost before calling this function.

        Parameters
        ----------
        pc : CarrayContainer
            Particle data
        """
        # container should not have ghost particles
        cdef int num_real_particles = particles.get_number_of_items()

        self.set_radius(particles, num_real_particles)
        self.boundary_condition.create_ghost_particles_serial(
                particles,
                self.domain,
                num_real_particles)

        # container should now have ghost particles
        return particles.get_number_of_items() - num_real_particles

#    cdef create_ghost_particles_parallel(self, CarrayContainer particles):
#        """
#        Main routine in creating ghost particles. This routine works only in
#        parallel. It is responsible in creating interior and exterior ghost
#        particles. Once finished buffer_ids and buffer_pids will have the
#        order of ghost particles sent to other processors. Particle container
#        should not have any prior ghost before calling this function.
#
#        Parameters
#        ----------
#        pc : CarrayContainer
#            Particle data
#        """
#        cdef CarrayContainer ghost
#
#        cdef int incoming_ghost
#        cdef np.ndarray ind, buf_pid, buf_ids
#        cdef int num_real_particles = pc.get_number_of_items()
#
#        # clear out all buffer arrays
#        self.buffer_ids.clear()
#
#        # flag ghost to be sent to other processors
#        # buffer arrays are now populated
#        self.create_interior_ghost_particles(particles, ghost)
#        self.boundary_condition.create_ghost_particles(particles,
#                ghost, self.load_bal, self.buffer_ids)
#
#        self.start_ghost = particles.get_number_of_items()
#
#        qsort(self.buffer_ids.begin(), self.buffer_size.end(), proc_compare)
#
#        # copy inices and setup send and receives
#        self.indices.resize(self.buffer_size)
#        for i in range(self.buffer_ids.size()):
#            self.indices.data[i] = self.buffer_ids[i].index
#            self.send_cnts[self.buffer_ids[i].proc] += 1
#
#        # how many particles are you receiving from each processor
#        self.recv_particles[:] = 0
#        self.comm.Alltoall(sendbuf=self.send_cnts, recvbuf=self.recv_cnts)
#        incoming_ghost = np.sum(self.recv_cnts)
#
#        # make room for new ghost particles
#        particles.extend(incoming_ghost)
#
#        # import ghost particles and add to container
#        exchange_particles(particles, ghost, self.send_particles, self.recv_particles,
#                self.start_ghost, self.comm)
#
#        return pc.get_number_of_items() - num_real_particles
#
#    cdef CarrayContainer _create_interior_ghost_particles(self, CarrayContainer pc, int num_real_particles):
#        """
#        Create ghost particles from neighboring processor. Should only be used
#        in parallel runs. This routine modifies the radius of each particle.
#
#        Parameters
#        ----------
#        pc : CarrayContainer
#            Particle data
#        """
#        cdef DoubleArray r = pc.get_carray("radius")
#        cdef LongLongArray keys = pc.get_carray("key")
#
#        cdef CarrayContainer interior_ghost
#
#        cdef Node* node
#        cdef Tree glb_tree = self.load_bal.tree
#
#        cdef LongArray leaf_pid = self.load_bal.leaf_pid
#        cdef LongArray nbrs_pid = LongArray()
#        cdef np.ndarray nbrs_pid_npy
#
#        cdef np.float64_t *x[3]
#        cdef double xp[3], fac = self.scale_factor
#
#        cdef int i, j, dim = self.domain.dim
#
#        pc.pointer_groups(x, pc.named_groups['position'])
#
#        for i in range(num_real_particles):
#
#            # find size of leaf where particle lives in global tree
#            node = glb_tree.find_leaf(keys.data[i])
#            r.data[i] = min(fac*node.box_length/glb_tree.domain_fac, r.data[i])
#
#            for j in range(dim):
#                xp[j] = x[j][i]
#
#            # find overlaping processors
#            nbrs_pid.reset()
#            glb_tree.get_nearest_process_neighbors(
#                    xp, r.data[i], leaf_pid, self.rank, nbrs_pid)
#
#            if nbrs_pid.length:
#
#                # put in processors in order to avoid duplicates
#                nbrs_pid_npy = nbrs_pid.get_npy_array()
#                nbrs_pid_npy.sort()
#
#                # put particles in buffer
#                self.buffer_ids.append(i)               # store particle id
#                self.buffer_pid.append(nbrs_pid_npy[0]) # store export processor
#
#                for j in range(1, nbrs_pid.length):
#
#                    if nbrs_pid_npy[j] != nbrs_pid_npy[j-1]:
#                        self.buffer_ids.append(i)               # store particle id
#                        self.buffer_pid.append(nbrs_pid_npy[j]) # store export processor
#
#        interior_ghost = pc.extract_items(self.buffer_ids)
#
#        interior_ghost["tag"][:] = Ghost
#        interior_ghost["type"][:] = Interior
#
#        return interior_ghost

    def migrate_boundary_particles(self, CarrayContainer particles):
        '''
        For moving mesh simulations particles may have moved outside
        the domain or processor patch after integration step. This function
        sends particles to their respective locations.
        '''
#        if phd._in_parallel:
#            self.migrate_boundary_particles_parallel(particles)
#        else:
        self.migrate_boundary_particles_serial(particles)

    def migrate_boundary_particles_serial(self, CarrayContainer particles):
        """
        After a simulation timestep in a serial run, particles may have left the domain.
        This routine ensures the particles are kept in the domain.

        Parameters
        ----------
        particles : CarrayContainer
            Particle data
        """
        cdef IntArray flags = particles.get_carray("flag")

        cdef double xp[3], *x[3]
        cdef int i, j, is_outside
        cdef int dim = particles.info['dim']

        particles.remove_tagged_particles(ParticleTAGS.Ghost)
        particles.pointer_groups(x, particles.named_groups['position'])

        for i in range(particles.get_number_of_items()):

            # did particle leave domain
            is_outside = 0
            for j in range(dim):
                xp[j] = x[j][i]
                is_outside += xp[j] <= self.domain.bounds[0][j] or self.domain.bounds[1][j] <= xp[j]

            if is_outside: # particle left domain
                if(self.boundary_types & REFLECTIVE):
                    raise RuntimeError("particle left domain in reflective boundary condition!!")

                elif(self.boundary_types & PERIODIC):

                    # wrap particle back in domain
                    for j in range(dim):
                        if xp[j] <= self.domain.bounds[0][j]:
                            x[j][i] += self.domain.translate[j]
                        if xp[j] >= self.domain.bounds[1][j]:
                            x[j][i] -= self.domain.translate[j]

#    cdef migrate_boundary_particles_parallel(self, CarrayContainer particles):
#        """
#        After a simulation timestep in a parallel run, particles may have left processor patch.
#        This routine export all particles that have left.
#
#        Parameters
#        ----------
#        pc : CarrayContainer
#            Particle data
#        """
#        cdef Tree tree
#        cdef Node *node
#
#        cdef IntArray tags
#        cdef LongLongArray keys
#        cdef CarrayContainer export_pc
#
#        cdef LongArray leaf_pid
#
#        cdef int dim = self.domain.dim
#
#        cdef double fac
#        cdef np.int32_t xh[3]
#        cdef double xp[3], *x[3]
#        cdef int i, j, is_outside, incoming_ghost, pid
#
#        cdef np.ndarray corner
#        cdef np.ndarray buf_pid, buf_ids
#
#        # buffers need to be locked
#        particles.remove_tagged_particles(ParticleTAGS.Ghost)
#        self.buffer_ids.reset()
#        self.buffer_pid.reset()
#
#        # information to map coordinates to hilbert space
#        fac = self.load_bal.fac
#        corner = self.load_bal.corner
#        leaf_pid = self.load_bal.leaf_pid
#
#        # reference to global tree
#        glb_tree = self.load_bal.tree
#
#        flags = particles.get_carray("flag")
#        keys = particles.get_carray("key")
#        particles.pointer_groups(x, particles.named_groups['position'])
#
#        for i in range(particles.get_number_of_items()):
#            if flags.data[i] == REAL:
#
#               # did the particle leave the domain
#                is_outside = 0
#                for j in range(dim):
#                    xp[j] = x[j][i]
#
#                    # check if particle left any dimension
#                    is_outside += xp[j] <= self.domain.bounds[0][j] or\
#                            self.domain.bounds[1][j] <= xp[j]
#
#                if is_outside:
#
#                    # should not happen in reflective case
#                    if (# ?? # & REFLECTIVE):
#                        raise RuntimeError("particle left domain in reflective boundary condition!!")
#
#                    # wrap particle back in domain
#                    elif (# ?? # & PERIODIC):
#                        for j in range(dim):
#
#                            # left min bound
#                            if xp[j] <= self.domain.bounds[0][j]:
#                                x[j][i] += self.domain.translate[j]
#
#                            # left max bound
#                            if xp[j] >= self.domain.bounds[1][j]:
#                                x[j][i] -= self.domain.translate[j]
#
#                # new hilbert key
#                for j in range(dim):
#                    xh[j] = <np.int32_t> ( (x[j][i] - corner[j])*fac )
#
#                # update particle hilberts key
#                keys.data[i] = self.load_bal.hilbert_func(xh[0], xh[1], xh[2],
#                        self.load_bal.order)
#
#                # find which processor particle lives in
#                node = glb_tree.find_leaf(keys.data[i])
#                pid  = leaf_pid.data[node.array_index]
#
#                # has particle left the domain
#                if pid != self.rank:
#                    self.buffer_ids.push_back(PairId(i, pid))
#
#        for i in range(self.size):
#            self.send_cnts[i] = 0
#            self.recv_cnts[i] = 0
#
#        if self.buffer_ids.size():
#
#            # organize particle processors
#            qsort(self.buffer_ids.begin(), self.buffer_size.end(), proc_compare)
#
#            # copy inices and setup send and receives
#            self.indices.resize(self.buffer_size)
#            for i in range(self.buffer_ids.size()):
#                self.indices.data[i] = self.buffer_ids[i].index
#                self.send_cnts[self.buffer_ids[i].proc] += 1
#
#            # copy flagged particles and remove form particles
#            particles_export = particles.extract_items(self.indices)
#            particles.remove_items(self.indices)
#
#        else:
#
#            # if no particles flagged still participate, particles can be incoming
#            particles_export = CarrayContainer(var_dict=particles.carray_info)
#
#        # place new particles starting at this index
#        begin_index = particles.get_number_of_items()
#
#        # how many particles are you receiving from each processor
#        self.comm.Alltoall(sendbuf=self.send_cnts, recvbuf=self.recv_cnts)
#        incoming_particles = np.sum(self.recv_cnts)
#
#        # make room for new ghost particles
#        particles.extend(incoming_particles)
#
#        # add new particles to container 
#        exchange_particles(particles, particles_export, self.send_cnts, self.recv_cnts,
#                begin_index, self.comm)
#
#    cdef _update_gradients(self, CarrayContainer particles, CarrayContainer gradient, list fields):
#        """
#        Transfer gradient from image particle to ghost particle.
#
#        Parameters
#        ----------
#        pc : CarrayContainer
#            Gradient data
#        fields : list
#            List of field strings to update
#        """
#        cdef int i, j, ip, dim = self.domain.dim
#        cdef np.float64_t *x[3], *dv[4]
#
#        cdef LongArray indices = LongArray()
#        cdef np.ndarray indices_npy, map_indices_npy
#        cdef IntArray types = particles.get_carray("type")
#
#        # find all ghost that need to be updated
#        for i in range(particles.get_number_of_items()):
#            if types.data[i] == Exterior:
#                indices.append(i)
#
#        indices_npy = indices.get_npy_array()
#        map_indices_npy = particles["map"][indices_npy]
#
#        # update ghost with their image data
#        for field in fields:
#            gradient[field][indices_npy] = gradient[field][map_indices_npy]
#
#        # refective bc, mirror velocity gradients
#        if self.boundary_type == BoundaryType.Reflective:
#
#            pc.pointer_groups(x, particles.named_groups['position'])
#            gradient.pointer_groups(dv, gradient.named_groups['velocity'])
#
#            for i in range(indices_npy.size): # every exterior ghost
#                for j in range(dim): # each dimension
#
#                    ip = indices.data[i]
#
#                    # reverse the normal of velocity gradient
#                    if x[j][ip] < self.domain.bounds[0][j]:
#                        # flip gradient component
#                        dv[(dim+1)*j][ip] *= -1
#
#                    if x[j][ip] > self.domain.bounds[1][j]:
#                        # flip gradient component
#                        dv[(dim+1)*j][ip] *= -1
#
    cdef values_to_ghost(CarrayContainer particles, list fields):
        if phd._in_parallel:
            self.values_to_ghost_serial(particles, fields)
        else:
            self.values_to_ghost_parallel(particles, fields)

#    cdef _update_ghost_particles(self, CarrayContainer particles, list fields):
#        """
#        Transfer data from image particle to ghost particle. Works only in
#        parallel.
#
#        Parameters
#        ----------
#        pc : CarrayContainer
#            Particle data
#        fields : dict
#            List of field strings to update
#        """
#        cdef str field
#        cdef CarrayContainer ghost
#
#        # reflective is a special case
#        if self.boundary_type == BoundaryType.Reflective:
#            Boundary._update_ghost_particles(self, pc, fields)
#
#        ghost = particles.extract_items(self.indices, fields)
#        exchange_particles(particles, ghost, self.send_particles, self.recv_particles,
#                self.start_ghost, self.comm, fields)
