import numpy as np

from ..hilbert.hilbert cimport hilbert_key_2d, hilbert_key_3d
from ..utils.particle_tags import ParticleTAGS
from ..utils.exchange_particles import exchange_particles
from libcpp.vector cimport vector

cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost
cdef int Exterior = ParticleTAGS.Exterior
cdef int Interior = ParticleTAGS.Interior

cdef class BoundaryType:
    Reflective = 0
    Periodic = 1

cdef int in_box(np.float64_t x[3], np.float64_t r, np.float64_t bounds[2][3], int dim):
    """
    Check if particle bounding box overlaps with a box defined by bounds.

    Parameters
    ----------
    x : array[3]
        Particle position
    r : np.float64_t
        Particle radius
    bounds : array[2][3]
        min/max of bounds in each dimension
    dim : int
        Problem dimension
    """
    cdef int i
    for i in range(dim):
        if x[i] + r < bounds[0][i]:
            return 0
        if x[i] - r > bounds[1][i]:
            return 0
    return 1

cdef _reflective(ParticleContainer pc, DomainLimits domain, int num_real_particles):
    """
    Create reflective ghost particles in the simulation. Can be used in non-parallel
    and parallel runs. Ghost particles are appended right after real particles in
    the container. Particle container should only have real particles when used.

    Parameters
    ----------
    pc : ParticleContainer
        Particle data
    domain : DomainLimits
        Information of the domain size and coordinates
    num_real_particles : int
        Number of real particles in the container
    """
    cdef CarrayContainer exterior_ghost

    cdef DoubleArray r = pc.get_carray("radius")

    cdef IntArray tags
    cdef IntArray types
    cdef LongArray maps

    cdef vector[Particle] ghost_particle
    cdef Particle *p

    cdef double xp[3], vp[3]
    cdef np.float64_t *x[3], *v[3]
    cdef np.float64_t *xg[3], *vg[3]

    cdef int i, j, k
    cdef dim = domain.dim
    cdef LongArray indices = LongArray()

    pc.extract_field_vec_ptr(x, "position")
    pc.extract_field_vec_ptr(v, "velocity")

    for i in range(num_real_particles):
        for j in range(dim):

            # lower boundary
            # does particle radius leave global boundary 
            if x[j][i] - r.data[i] < domain.bounds[0][j]:

                # copy particle information
                for k in range(dim):
                    xp[k] = x[k][i]
                    vp[k] = v[k][i]

                # reflect particle position and velocity 
                xp[j] =  x[j][i] - 2*(x[j][i] - domain.bounds[0][j])
                vp[j] = -v[j][i]

                # store new ghost position/velocity and image index
                ghost_particle.push_back(Particle(xp, vp, dim))
                indices.append(i)

            # upper boundary
            # does particle radius leave global boundary 
            if domain.bounds[1][j] < x[j][i] + r.data[i]:

                # copy particle information
                for k in range(dim):
                    xp[k] = x[k][i]
                    vp[k] = v[k][i]

                # reflect particle position and velocity
                xp[j] =  x[j][i] - 2*(x[j][i] - domain.bounds[1][j])
                vp[j] = -v[j][i]

                # store new ghost position/velocity and image index
                ghost_particle.push_back(Particle(xp, vp, dim))
                indices.append(i)

    # in parallel a patch might not have exterior ghost
    if indices.length:

        # create ghost particles from flagged particles
        exterior_ghost = pc.extract_items(indices.get_npy_array())

        # references to new ghost data
        exterior_ghost.extract_field_vec_ptr(xg, "position")
        exterior_ghost.extract_field_vec_ptr(vg, "velocity")

        maps = exterior_ghost.get_carray("map")
        tags = exterior_ghost.get_carray("tag")
        types = exterior_ghost.get_carray("type")

        # transfer new data to ghost 
        for i in range(exterior_ghost.get_number_of_items()):

            maps.data[i] = indices.data[i]  # reference to image
            tags.data[i] = Ghost            # ghost label
            types.data[i] = Exterior        # exterior ghost label

            # update new position/velocity
            p = &ghost_particle[i]
            for j in range(dim):
                xg[j][i] = p.x[j]
                vg[j][i] = p.v[j]

        # add new ghost to particle container
        pc.append_container(exterior_ghost)

cdef _periodic(ParticleContainer pc, DomainLimits domain, int num_real_particles):
    """
    Create periodic ghost particles in the simulation. Should only be used in
    non-parallel runs. Ghost particles are appended right after real particles in
    the container. Particle container should only have real particles when used.

    Parameters
    ----------
    pc : ParticleContainer
        Particle data
    domain : DomainLimits
        Information of the domain size and coordinates
    num_real_particles : int
        Number of real particles in the container
    """
    cdef CarrayContainer exterior_ghost

    cdef DoubleArray r = pc.get_carray("radius")

    cdef LongArray maps
    cdef IntArray tags
    cdef IntArray types

    cdef vector[Particle] ghost_particle
    cdef Particle *p

    cdef double xp[3], vp[3]
    cdef double xs[3]
    cdef np.float64_t *x[3]
    cdef np.float64_t *xg[3]

    cdef int dim = domain.dim
    cdef int i, j, k, index[3]
    cdef int num_shifts = dim**3
    cdef LongArray indices = LongArray()

    pc.extract_field_vec_ptr(x, "position")

    # velocities not needed set to zero
    for j in range(dim):
        vp[j] = 0

    for i in range(num_real_particles):
        for j in range(dim):
            xp[j] = x[j][i]

        # check if particle intersects global boundary
        if in_box(xp, r.data[i], domain.bounds, dim):

            # shift particle coordinates to create periodic ghost
            for k in range(num_shifts):

                # shift indices
                index[0] = k%3; index[1] = (k/3)%3; index[2] = k/9

                # skip no shift
                if (k == 4 and dim == 2) or (k == 13 and dim == 3):
                    continue

                # shifted position
                for j in range(dim):
                    xs[j] = xp[j] + (index[j]-1)*domain.translate[j]

                # check if new ghost intersects global boundary
                if in_box(xs, r.data[i], domain.bounds, dim):
                    ghost_particle.push_back(Particle(xs, vp, dim))
                    indices.append(i)

    # create ghost particles from flagged particles
    exterior_ghost = pc.extract_items(indices.get_npy_array())

    # references to new ghost data
    exterior_ghost.extract_field_vec_ptr(xg, "position")

    maps = exterior_ghost.get_carray("map")
    tags = exterior_ghost.get_carray("tag")
    types = exterior_ghost.get_carray("type")

    # transfer data to ghost particles and update labels
    for i in range(exterior_ghost.get_number_of_items()):

        maps.data[i]  = indices.data[i]
        tags.data[i]  = Ghost
        types.data[i] = Exterior

        # only position change in periodic conditions
        p = &ghost_particle[i]
        for j in range(dim):
            xg[j][i] = p.x[j]

    # add periodic ghost to particle container
    pc.append_container(exterior_ghost)

cdef _periodic_parallel(ParticleContainer pc, CarrayContainer ghost, DomainLimits domain,
        Tree glb_tree, LongArray leaf_pid, LongArray buffer_ids, LongArray buffer_pid,
        int num_real_particles, int rank):
    """
    Create periodic ghost particles in the simulation. Should only be used in
    parallel runs. Ghost particles are added to ghost container not particle
    container. The ids used to create the ghost particles are stored in buffer_ids
    and the processors destination for the ghost are stored in buffer_pid. Unlike the
    non-parallel periodic the particle container is not modified instead ghost particles
    are put in ghost containers and buffers are modified.

    Parameters
    ----------
    pc : ParticleContainer
        Particle data
    ghost : ParticleContainer
        Ghost data that will be exported to other processors
    domain : DomainLimits
        Information of the domain size and coordinates
    tree : Tree
        Global tree used in load balance, used for searches
    leaf_pid : LongArray
        Each leaf has an index to leaf_pid, value is processor id where leaf lives in
    buffer_ids : LongArray
        The local id used to create the ghost particle
    buffer_pids : LongArray
        Processor destination where ghost will be exported to
    num_real_particles : int
        Number of real particles in the container
    rank : int
        Processor rank
    """
    cdef DoubleArray r = pc.get_carray("radius")

    cdef np.ndarray nbrs_pid_npy
    cdef LongArray nbrs_pid = LongArray()

    cdef LongArray indices = LongArray()

    cdef CarrayContainer exterior_ghost
    cdef IntArray tags
    cdef IntArray types

    cdef vector[Particle] ghost_particle
    cdef Particle *p

    cdef double xp[3], vp[3]
    cdef double xs[3]
    cdef np.float64_t* x[3]
    cdef np.float64_t* xg[3]

    cdef int dim = domain.dim
    cdef int num_shifts = 3**dim
    cdef int i, j, m, num_neighbors, index[3]

    pc.extract_field_vec_ptr(x, "position")

    # velocities not needed set to zero
    for j in range(dim):
        vp[j] = 0

    for i in range(num_real_particles):
        for j in range(dim):
            xp[j] = x[j][i]

        # check if particle intersects global domain
        if in_box(xp, r.data[i], domain.bounds, dim):

            # shift particle coordinates to create periodic ghost
            for k in range(num_shifts):

                # create shift indices
                index[0] = k%3; index[1] = (k/3)%3; index[2] = k/9

                # skip no shift
                if (k == 4 and dim == 2) or (k == 13 and dim == 3):
                    continue

                # shifted position
                for j in range(dim):
                    xs[j] = xp[j] + (index[j]-1)*domain.translate[j]

                # check if new ghost intersects global boundary
                if in_box(xs, r.data[i], domain.bounds, dim):

                    # find neighboring processors
                    nbrs_pid.reset()
                    glb_tree.get_nearest_process_neighbors(
                            xs, r.data[i], leaf_pid, rank, nbrs_pid)

                    if nbrs_pid.length != 0:

                        # put in processors in order to avoid duplicates
                        nbrs_pid_npy = nbrs_pid.get_npy_array()
                        nbrs_pid_npy.sort()

                        # put particle in buffer
                        indices.append(i)                   # store for copying
                        buffer_ids.append(i)                # store particle id
                        buffer_pid.append(nbrs_pid_npy[0])  # store export processor id
                        ghost_particle.push_back(Particle(xs, vp, dim))

                        for j in range(1, nbrs_pid.length):

                            if nbrs_pid_npy[j] != nbrs_pid_npy[j-1]: # avoid duplicates
                                indices.append(i)                    # store for copying 
                                buffer_ids.append(i)                 # store particle id
                                buffer_pid.append(nbrs_pid_npy[j])   # store export processor id
                                ghost_particle.push_back(Particle(xs, vp, dim))

    # patch might not have exterior ghost
    if indices.length:

        # create ghost particles from flagged particles
        exterior_ghost = pc.extract_items(indices.get_npy_array())

        # references to new ghost data
        exterior_ghost.extract_field_vec_ptr(xg, "position")
        tags = exterior_ghost.get_carray("tag")
        types = exterior_ghost.get_carray("type")

        # transfer data to ghost particles
        for i in range(exterior_ghost.get_number_of_items()):

            tags.data[i] = Ghost
            types.data[i] = Exterior

            p = &ghost_particle[i]
            for j in range(dim):
                xg[j][i] = p.x[j]

        # add periodic ghost to ghost container
        ghost.append_container(exterior_ghost)

cdef class Boundary:
    def __init__(self, DomainLimits domain, int boundary_type):

        self.domain = domain
        self.boundary_type = boundary_type

    cdef _set_radius(self, ParticleContainer pc, int num_real_particles):
        """
        Filter particle radius. This is done because in the initial mesh creation
        some particles will have infinite radius.

        Parameters
        ----------
        pc : ParticleContainer
            Particle data
        num_real_particles : int
            Number of real particles in the container
        """
        cdef int i
        cdef DoubleArray r = pc.get_carray("radius")
        cdef double box_size = self.domain.max_length

        for i in range(num_real_particles):
            r.data[i] = min(0.4*box_size, r.data[i])

    cdef int _create_ghost_particles(self, ParticleContainer pc):
        """
        Main routine in creating ghost particles. This routine appends ghost
        particles in particle container. Particle container should not have
        any prior ghost before calling this function.

        Parameters
        ----------
        pc : ParticleContainer
            Particle data
        """
        # container should not have ghost particles
        cdef int num_real_particles = pc.get_number_of_particles()

        self._set_radius(pc, num_real_particles)
        if self.boundary_type == BoundaryType.Reflective:
            _reflective(pc, self.domain, num_real_particles)
        if self.boundary_type == BoundaryType.Periodic:
            _periodic(pc, self.domain, num_real_particles)

        # container should now have ghost particles
        return pc.get_number_of_particles() - num_real_particles

    cdef _update_ghost_particles(self, ParticleContainer pc, list fields):
        """
        Transfer data from image particle to ghost particle.

        Parameters
        ----------
        pc : ParticleContainer
            Particle data
        fields : list
            List of field strings to update
        """
        cdef LongArray indices = LongArray()
        cdef IntArray types = pc.get_carray("type")
        cdef np.ndarray indices_npy, map_indices_npy

        # find all ghost that need to be updated
        for i in range(pc.get_number_of_particles()):
            if types.data[i] == Exterior:
                indices.append(i)

        indices_npy = indices.get_npy_array()
        map_indices_npy = pc["map"][indices_npy]

        # update ghost with their image data
        for field in fields:
            pc[field][indices_npy] = pc[field][map_indices_npy]

    def migrate_boundary_particles(self, ParticleContainer pc):
        """
        After a simulation timestep in a parallel run, particles may have left processor patch.
        This routine export all particles that have left.

        Parameters
        ----------
        pc : ParticleContainer
            Particle data
        """
        cdef IntArray tags

        cdef int dim = self.domain.dim
        cdef double xp[3], *x[3]
        cdef int i, j, is_outside

        pc.remove_tagged_particles(ParticleTAGS.Ghost)
        tags = pc.get_carray("tag")
        pc.extract_field_vec_ptr(x, "position")

        for i in range(pc.get_number_of_particles()):

            # did particle leave domain
            is_outside = 0
            for j in range(dim):
                xp[j] = x[j][i]
                is_outside += xp[j] <= self.domain.bounds[0][j] or self.domain.bounds[1][j] <= xp[j]

            if is_outside: # particle left domain
                if self.boundary_type == BoundaryType.Reflective:
                    raise RuntimeError("particle left domain in reflective boundary condition!!")
                elif self.boundary_type == BoundaryType.Periodic:

                    # wrap particle back in domain
                    for j in range(dim):
                        if xp[j] <= self.domain.bounds[0][j]:
                            x[j][i] += self.domain.translate[j]
                        if xp[j] >= self.domain.bounds[1][j]:
                            x[j][i] -= self.domain.translate[j]

    def flag_real_and_ghost(self, ParticleContainer pc):

        cdef IntArray tags
        cdef np.float64_t *x[3]
        cdef int i, j, is_outside
        cdef int dim = self.domain.dim

        pc.remove_tagged_particles(ParticleTAGS.Ghost)
        tags = pc.get_carray("tag")
        pc.extract_field_vec_ptr(x, "position")

        for i in range(pc.get_number_of_particles()):

            # did particle leave domain
            is_outside = 0
            for j in range(dim):
                is_outside += x[j][i] <= self.domain.bounds[0][j] or self.domain.bounds[1][j] <= x[j][i]

            if is_outside:  # particle left domain
                tags.data[i] = Ghost
            else:           # particle remain domain
                tags.data[i] = Real

cdef class BoundaryParallel(Boundary):
    def __init__(self, DomainLimits domain, int boundary_type, LoadBalance load_bal, object comm):

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.load_bal = load_bal

        self.recv_particles = np.empty(self.size, np.int32)
        self.send_particles = np.empty(self.size, np.int32)

        self.buffer_ids = LongArray()
        self.buffer_pid = LongArray()

        self.hilbert_func = NULL

        self.domain = domain
        self.boundary_type = boundary_type

        if domain.dim == 2:
            self.hilbert_func = hilbert_key_2d
        elif domain.dim == 3:
            self.hilbert_func = hilbert_key_3d
        else:
            raise RuntimeError("Wrong dimension for tree")

    cdef CarrayContainer _create_interior_ghost_particles(self, ParticleContainer pc, int num_real_particles):
        """
        Create ghost particles from neighboring processor. Should only be used
        in parallel runs. This routine modifies the radius of each particle.

        Parameters
        ----------
        pc : ParticleContainer
            Particle data
        """
        cdef DoubleArray r = pc.get_carray("radius")
        cdef LongLongArray keys = pc.get_carray("key")

        cdef CarrayContainer interior_ghost

        cdef Node* node
        cdef Tree glb_tree = self.load_bal.tree

        cdef LongArray leaf_pid = self.load_bal.leaf_pid
        cdef LongArray nbrs_pid = LongArray()
        cdef np.ndarray nbrs_pid_npy

        cdef np.float64_t *x[3]
        cdef double xp[3]

        cdef int i, j, dim = self.domain.dim

        pc.extract_field_vec_ptr(x, "position")

        for i in range(num_real_particles):

            # find size of leaf where particle lives in global tree
            node = glb_tree.find_leaf(keys.data[i])
            r.data[i] = min(0.5*node.box_length/glb_tree.domain_fac, r.data[i])

            for j in range(dim):
                xp[j] = x[j][i]

            # find overlaping processors
            nbrs_pid.reset()
            glb_tree.get_nearest_process_neighbors(
                    xp, r.data[i], leaf_pid, self.rank, nbrs_pid)

            if nbrs_pid.length:

                # put in processors in order to avoid duplicates
                nbrs_pid_npy = nbrs_pid.get_npy_array()
                nbrs_pid_npy.sort()

                # put particles in buffer
                self.buffer_ids.append(i)               # store particle id
                self.buffer_pid.append(nbrs_pid_npy[0]) # store export processor

                for j in range(1, nbrs_pid.length):

                    if nbrs_pid_npy[j] != nbrs_pid_npy[j-1]:
                        self.buffer_ids.append(i)               # store particle id
                        self.buffer_pid.append(nbrs_pid_npy[j]) # store export processor

        interior_ghost = pc.extract_items(self.buffer_ids.get_npy_array())

        interior_ghost["tag"][:] = Ghost
        interior_ghost["type"][:] = Interior

        return interior_ghost

    cdef int _create_ghost_particles(self, ParticleContainer pc):
        """
        Main routine in creating ghost particles. This routine works only in
        parallel. It is responsible in creating interior and exterior ghost
        particles. Once finished buffer_ids and buffer_pids will have the
        order of ghost particles sent to other processors. Particle container
        should not have any prior ghost before calling this function.

        Parameters
        ----------
        pc : ParticleContainer
            Particle data
        """
        cdef CarrayContainer ghost

        cdef np.ndarray ind, buf_pid, buf_ids

        cdef int incoming_ghost
        cdef int num_real_particles = pc.get_number_of_particles()

        # clear out all buffer arrays
        self.buffer_ids.reset()
        self.buffer_pid.reset()

        # flag ghost to be sent to other processors
        # buffer arrays are now populated
        ghost = self._create_interior_ghost_particles(pc, num_real_particles)

        # create global boundary ghost 
        if self.boundary_type == BoundaryType.Reflective:
            _reflective(pc, self.domain, num_real_particles)
            # reflective ghost are appendend to pc
            self.start_ghost = pc.get_number_of_particles()
        if self.boundary_type == BoundaryType.Periodic:
            _periodic_parallel(pc, ghost, self.domain, self.load_bal.tree,
                    self.load_bal.leaf_pid, self.buffer_ids, self.buffer_pid,
                    num_real_particles, self.rank)
            # periodic ghost are not appendend to pc
            self.start_ghost = num_real_particles

        # use numpy arrays for convience
        buf_pid = self.buffer_pid.get_npy_array()
        buf_ids = self.buffer_ids.get_npy_array()

        # organize particle processors
        ind = np.argsort(buf_pid)
        buf_ids[:] = buf_ids[ind]
        buf_pid[:] = buf_pid[ind]

        # do for fields too
        for field in ghost.properties.keys():
            ghost[field][:] = ghost[field][ind]

        # bin the number of particles being sent to each processor
        self.send_particles[:] = np.bincount(buf_pid,
                minlength=self.size).astype(np.int32)

        # how many particles are you receiving from each processor
        self.recv_particles[:] = 0
        self.comm.Alltoall(sendbuf=self.send_particles, recvbuf=self.recv_particles)
        incoming_ghost = np.sum(self.recv_particles)

        # make room for new ghost particles
        pc.extend(incoming_ghost)

        # import ghost particles and add to container
        exchange_particles(pc, ghost, self.send_particles, self.recv_particles,
                self.start_ghost, self.comm)

        return pc.get_number_of_particles() - num_real_particles

    cdef _update_ghost_particles(self, ParticleContainer pc, list fields):
        """
        Transfer data from image particle to ghost particle. Works only in
        parallel.

        Parameters
        ----------
        pc : ParticleContainer
            Particle data
        fields : dict
            List of field strings to update
        """
        cdef str field
        cdef CarrayContainer ghost

        # reflective is a special case
        if self.boundary_type == BoundaryType.Reflective:
            Boundary._update_ghost_particles(self, pc, fields)

        ghost = pc.extract_items(self.buffer_ids.get_npy_array(), fields)
        exchange_particles(pc, ghost, self.send_particles, self.recv_particles,
                self.start_ghost, self.comm, fields)

    def migrate_boundary_particles(self, ParticleContainer pc):
        """
        After a simulation timestep in a parallel run, particles may have left processor patch.
        This routine export all particles that have left.

        Parameters
        ----------
        pc : ParticleContainer
            Particle data
        """
        cdef Tree tree
        cdef Node *node

        cdef IntArray tags
        cdef LongLongArray keys
        cdef CarrayContainer export_pc

        cdef LongArray leaf_pid

        cdef int dim = self.domain.dim

        cdef double fac
        cdef np.int32_t xh[3]
        cdef double xp[3], *x[3]
        cdef int i, j, is_outside, incoming_ghost, pid

        cdef np.ndarray corner
        cdef np.ndarray buf_pid, buf_ids

        # buffers need to be locked
        pc.remove_tagged_particles(ParticleTAGS.Ghost)
        self.buffer_ids.reset()
        self.buffer_pid.reset()

        # information to map coordinates to hilbert space
        fac = self.load_bal.fac
        corner = self.load_bal.corner
        leaf_pid = self.load_bal.leaf_pid

        # reference to global tree
        glb_tree = self.load_bal.tree

        tags = pc.get_carray("tag")
        keys = pc.get_carray("key")
        pc.extract_field_vec_ptr(x, "position")

        for i in range(pc.get_number_of_particles()):
            if tags.data[i] == Real:

               # did the particle leave the domain
                is_outside = 0
                for j in range(dim):
                    xp[j] = x[j][i]
                    is_outside += xp[j] <= self.domain.bounds[0][j] or self.domain.bounds[1][j] <= xp[j]

                if is_outside: # particle left domain
                    if self.boundary_type == BoundaryType.Reflective:
                        raise RuntimeError("particle left domain in reflective boundary condition!!")
                    elif self.boundary_type == BoundaryType.Periodic:

                        # wrap particle back in domain
                        for j in range(dim):
                            if xp[j] <= self.domain.bounds[0][j]:
                                x[j][i] += self.domain.translate[j]
                            if xp[j] >= self.domain.bounds[1][j]:
                                x[j][i] -= self.domain.translate[j]

                # generate new hilbert key
                for j in range(dim):
                    xh[j] = <np.int32_t> ( (x[j][i] - corner[j])*fac )

                keys.data[i] = self.load_bal.hilbert_func(xh[0], xh[1], xh[2], self.load_bal.order)

                # find which processor particle lives in
                node = glb_tree.find_leaf(keys.data[i])
                pid  = leaf_pid.data[node.array_index]

                if pid != self.rank: # flag to export
                    self.buffer_ids.append(i)
                    self.buffer_pid.append(pid)

        # use numpy arrays for convience
        buf_pid = self.buffer_pid.get_npy_array()
        buf_ids = self.buffer_ids.get_npy_array()

        if buf_ids.size:

            # organize particle processors
            ind = np.argsort(buf_pid)
            buf_ids[:] = buf_ids[ind]
            buf_pid[:] = buf_pid[ind]

            export_pc = pc.extract_items(buf_ids)
            pc.remove_items(buf_ids)

            # bin the number of particles being sent to each processor
            self.send_particles[:] = np.bincount(buf_pid,
                    minlength=self.size).astype(np.int32)

        else:

            export_pc = ParticleContainer(var_dict=pc.carray_info)
            self.send_particles[:] = 0

        self.start_ghost = pc.get_number_of_particles()

        # how many particles are you receiving from each processor
        self.recv_particles[:] = 0
        self.comm.Alltoall(sendbuf=self.send_particles, recvbuf=self.recv_particles)
        incoming_ghost = np.sum(self.recv_particles)

        # make room for new ghost particles
        pc.extend(incoming_ghost)

        # import ghost particles and add to container
        exchange_particles(pc, export_pc, self.send_particles, self.recv_particles,
                self.start_ghost, self.comm)
