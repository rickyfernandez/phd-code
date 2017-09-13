import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from ..utils.particle_tags import ParticleTAGS
from ..utils.exchange_particles import exchange_particles
from ..hilbert.hilbert cimport hilbert_key_2d, hilbert_key_3d

cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost
cdef int Exterior = ParticleTAGS.Exterior
cdef int Interior = ParticleTAGS.Interior







class Periodic(BoundaryBase):
    def __init__(self):
        pass

    cdef bint create_ghost_particle_serial(ImageParticle *p, DomainManager domain_manager)
        """
        Create periodic ghost particles in the simulation. Should only be used in
        """
        cdef DoubleArray r = particles.get_carray("radius")

        cdef np.ndarray nbrs_pid_npy
        cdef LongArray nbrs_pid = LongArray()
        cdef LongArray indices = LongArray()


        cdef double xs[3]
        cdef np.float64_t* xg[3]

        cdef int dim = particles.info["dim"]
        cdef int num_shifts = 3**dim
        cdef int k, num_neighbors, index[3]

        # check if particle intersects global domain
        if in_box(p.x, p.new_radius, domain_manager.domain.bounds, dim):

            # shift particle coordinates to create periodic ghost
            for k in range(num_shifts):

                # create shift indices
                index[0] = k%3; index[1] = (k/3)%3; index[2] = k/9
                if (k == 4 and dim == 2) or (k == 13 and dim == 3):
                    # skip no shift
                    continue

                # shifted position
                for j in range(dim):
                    xs[j] = p.x[j] + (index[j]-1)*domain_manager.domain.translate[j]

                # find if shifted particle intersects
                if in_box(xs, p.new_radius, domain_manager.domain.bounds, dim):
                    if in_box(xs, p.new_radius, domain_manager.domain.bounds, dim):
                        ghost_particle.push_back(ImageParticle(xs, vp, pid, dim))
                        return True

    cdef create_ghost_particle_parallel(CarrayContainer pc, CarrayContainer ghost, DomainLimits domain,
            Tree glb_tree, LongArray leaf_pid, LongArray buffer_ids, LongArray buffer_pid,
            int num_real_particles, int rank):
        """
        Create periodic ghost particles in the simulation. Should only be used in
        """
        cdef DoubleArray r = particles.get_carray("radius")

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


        particles.pointer_groups(x, particles.named_groups['position'])

        # velocities not needed set to zero
        for j in range(dim):
            vp[j] = 0

        for j in range(dim):
            xp[j] = x[j][i]

        # check if particle intersects global domain
        if in_box(xp, r.data[i], domain.bounds, dim):

            # shift particle coordinates to create periodic ghost
            for k in range(num_shifts):

                # create shift indices
                index[0] = k%3; index[1] = (k/3)%3; index[2] = k/9
                if (k == 4 and dim == 2) or (k == 13 and dim == 3):
                    # skip no shift
                    continue

                # shifted position
                for j in range(dim):
                    xs[j] = xp[j] + (index[j]-1)*domain.translate[j]

                domain_manager.processor_intersection(
                        xs, radius1, radius2, nbrs, rank)

                if nbrs.size():
                    partice_fail = True
                    # this particle found more processors to be
                    # sent too
                    for j in range(nbrs_pid.length):

                        # store particle, particle id and processor
                        buffers.push_back(IdPair())
                        ghost_particle.push_back(Particle(xs, vp, dim))

        return particle_fail


cdef class BoundaryType:
    Reflective = 0
    Periodic = 1

cdef bint inline in_box(np.float64_t x[3], np.float64_t r, np.float64_t bounds[2][3], int dim):
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
            return False
        if x[i] - r > bounds[1][i]:
            return False
    return True

cdef inline flagged_processor(xs, r, ghost_particle, buffer_ids, buffer_pid,
        nbrs_pid, glb_tree, leaf_pid, rank):

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

cdef class BoundaryBase:
    cdef create_ghost_particles_serial(CarrayContainer particles, CarrayContainer ghost_particles,
            DomainManager domain_manager, int num_real_particles):
        """
        Evolve the simulation for one time step
        """
        msg = "IntegrateBase::evolve_timestep called!"
        raise NotImplementedError(msg)

    cdef create_ghost_particles_parallel(CarrayContainer particles, DomainLimits domain, int num_real_particles):
        """
        Evolve the simulation for one time step
        """
        msg = "IntegrateBase::evolve_timestep called!"
        raise NotImplementedError(msg)

cdef class Reflective(BoundaryBase):
    cdef create_ghost_particles_serial(CarrayContainer particles, CarrayContainer ghost_particles,
            DomainManager domain_manager, LongArray particles_indices):
        """
        Create reflective ghost particles in the simulation. Can be used in non-parallel
        and parallel runs. Ghost particles are appended right after real particles in
        the container. Particle container should only have real particles when used.

        Parameters
        ----------
        particles : CarrayContainer
            Particle data
        ghost_particles : CarrayContainer
            Ghost particle data
        domanin_manager : DomainManager
            Information of domain
        particles_indices : LongArray
            Indicies flagged for boundary ghost particles
        """
        cdef DoubleArray r = particles.get_carray("radius")

        cdef IntArray tags
        cdef IntArray types
        cdef LongArray maps
        cdef DoubleArray mass

        cdef Particle *p
        cdef vector[Particle] ghost_particle

        cdef double xp[3], vp[3]
        cdef np.float64_t *xg[3], *vg[3]
        cdef np.float64_t *x[3], *v[3], *mv[3]

        cdef int i, j, k
        cdef dim = domain.dim
        cdef LongArray indices = LongArray()

        particles.pointer_groups(x, particles.named_groups['position'])
        particles.pointer_groups(v, particles.named_groups['velocity'])

        exterior_ghost = particles.extract_items(particle_indices)

        for ip in range(particle_indices.size()):

            # extract real particle
            i = particle_indices[ip]
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
                    ghost_indices.append(i)

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
                    ghost_indices.append(i)

           # create ghost particles from flagged particles
        exterior_ghost = particles.extract_items(indices)

        # references to new ghost data
        exterior_ghost.pointer_groups(xg, particles.named_groups['position'])
        exterior_ghost.pointer_groups(vg, particles.named_groups['velocity'])

        maps = exterior_ghost.get_carray("map")
        tags = exterior_ghost.get_carray("tag")
        types = exterior_ghost.get_carray("type")

        mass = exterior_ghost.get_carray("mass")
        exterior_ghost.pointer_groups(mv, particles.named_groups['momentum'])

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
                mv[j][i] = mass.data[i]*p.v[j]

        # add new ghost to particle container
        ghost_particle.append_container(exterior_ghost)

    cdef create_ghost_particles_parallel(CarrayContainer particles, CarrayContainer ghost_particles,
        """
        Create reflective ghost particles in the simulation. Can be used in non-parallel
        and parallel runs. Ghost particles are appended right after real particles in
        the container. Particle container should only have real particles when used.

        Parameters
        ----------
        pc : CarrayContainer
            Particle data
        domain : DomainLimits
            Information of the domain size and coordinates
        num_real_particles : int
            Number of real particles in the container
        """
        cdef DoubleArray r = particles.get_carray("radius")

        cdef IntArray tags
        cdef IntArray types
        cdef LongArray maps
        cdef DoubleArray mass

        cdef Particle *p
        cdef vector[Particle] ghost_particle

        cdef double xp[3], vp[3]
        cdef np.float64_t *xg[3], *vg[3]
        cdef np.float64_t *x[3], *v[3], *mv[3]

        cdef int i, j, k
        cdef dim = domain.dim
        cdef LongArray indices = LongArray()

        particles.pointer_groups(x, particles.named_groups['position'])
        particles.pointer_groups(v, particles.named_groups['velocity'])

        for ip in range(particle_indices.size()):

            # extract real particle
            i = particle_indices[ip]
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

                    flagged_processor(xs, vp, r, ghost_particle, buffer_ids, buffer_pid,
                            nbrs_pid, glb_tree, leaf_pid, rank)

                    # store new ghost position/velocity and image index
                    ghost_particle.push_back(Particle(xp, vp, dim))
                    ghost_indices.append(i)

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
                    ghost_indices.append(i)

        # in parallel a patch might not have exterior ghost
        if ghost_indices.length:

           # create ghost particles from flagged particles
            exterior_ghost = particles.extract_items(indices)

            # references to new ghost data
            exterior_ghost.pointer_groups(xg, particles.named_groups['position'])
            exterior_ghost.pointer_groups(vg, particles.named_groups['velocity'])

            maps = exterior_ghost.get_carray("map")
            tags = exterior_ghost.get_carray("tag")
            types = exterior_ghost.get_carray("type")

            mass = exterior_ghost.get_carray("mass")
            exterior_ghost.pointer_groups(mv, pc.named_groups['momentum'])

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
                    mv[j][i] = mass.data[i]*p.v[j]

            # add new ghost to particle container
            particles.append_container(exterior_ghost)

    cdef create_ghost_particles_parallel(CarrayContainer particles, DomainLimits domain, int num_real_particles):
        self.create_ghost_particles_serial(particles, domain, num_real_particles)

    cdef _update_gradients(self, CarrayContainer pc, CarrayContainer gradient, list fields):
        """
        Transfer gradient from image particle to ghost particle.

        Parameters
        ----------
        pc : CarrayContainer
            Gradient data
        fields : list
            List of field strings to update
        """
        cdef int i, j, ip, dim = self.domain.dim
        cdef np.float64_t *x[3], *dv[4]

        cdef LongArray indices = LongArray()
        cdef IntArray types = pc.get_carray("type")
        cdef np.ndarray indices_npy, map_indices_npy

        # find all ghost that need to be updated
        for i in range(pc.get_number_of_items()):
            if types.data[i] == Exterior:
                indices.append(i)

        indices_npy = indices.get_npy_array()
        map_indices_npy = pc["map"][indices_npy]

        # update ghost with their image data
        for field in fields:
            gradient[field][indices_npy] = gradient[field][map_indices_npy]

        # refective bc, mirror velocity gradients
        if self.boundary_type == BoundaryType.Reflective:

            pc.pointer_groups(x, pc.named_groups['position'])
            gradient.pointer_groups(dv, gradient.named_groups['velocity'])

            for i in range(indices_npy.size): # every exterior ghost
                for j in range(dim): # each dimension

                    ip = indices.data[i]

                    # reverse the normal of velocity gradient
                    if x[j][ip] < self.domain.bounds[0][j]:
                        # flip gradient component
                        dv[(dim+1)*j][ip] *= -1

                    if x[j][ip] > self.domain.bounds[1][j]:
                        # flip gradient component
                        dv[(dim+1)*j][ip] *= -1


cdef class Periodic(BoundaryBase):
    cdef create_ghost_particles_serial(CarrayContainer particles, DomainLimits domain, int num_real_particles):
        """
        Create periodic ghost particles in the simulation. Should only be used in
        non-parallel runs. Ghost particles are appended right after real particles in
        the container. Particle container should only have real particles when used.

        Parameters
        ----------
        pc : CarrayContainer
            Particle data
        domain : DomainLimits
            Information of the domain size and coordinates
        num_real_particles : int
            Number of real particles in the container
        """
        cdef CarrayContainer exterior_ghost

        cdef DoubleArray r = particles.get_carray("radius")

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
        cdef int num_shifts = 3**dim
        cdef LongArray indices = LongArray()

        particles.pointer_groups(x, particles.named_groups['position'])

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
        exterior_ghost = pc.extract_items(indices)

        # references to new ghost data
        exterior_ghost.pointer_groups(xg, pc.named_groups['position'])

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

    cdef create_ghost_particles_parallel(CarrayContainer pc, CarrayContainer ghost, DomainLimits domain,
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
        pc : CarrayContainer
            Particle data
        ghost : CarrayContainer
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
        cdef DoubleArray r = particles.get_carray("radius")

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

        particles.pointer_groups(x, particles.named_groups['position'])

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
            exterior_ghost = particles.extract_items(indices)

            # references to new ghost data
            exterior_ghost.pointer_groups(xg, particles.named_groups['position'])
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
    def __init__(self, int boundary_type, double scale_factor=0.4, **kwargs):
        #self.domain = None
        self.scale_factor = scale_factor
        self.boundary_type = boundary_type

    cdef _set_radius(self, CarrayContainer pc, int num_real_particles):
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
        cdef DoubleArray r = pc.get_carray("radius")
        cdef double box_size = self.domain.max_length
        cdef double fac = self.scale_factor

        for i in range(num_real_particles):
            r.data[i] = min(fac*box_size, r.data[i])

    cdef _update_ghost_particles(self, CarrayContainer pc, list fields):
        """
        Transfer data from image particle to ghost particle.

        Parameters
        ----------
        pc : CarrayContainer
            Particle data
        fields : list
            List of field strings to update
        """
        cdef int i
        cdef LongArray indices = LongArray()
        cdef IntArray types = pc.get_carray("type")
        cdef np.ndarray indices_npy, map_indices_npy

        # find all ghost that need to be updated
        for i in range(pc.get_number_of_items()):
            if types.data[i] == Exterior:
                indices.append(i)

        if indices.length:

            indices_npy = indices.get_npy_array()
            map_indices_npy = pc["map"][indices_npy]

            # update ghost with their image data
            for field in fields:
                pc[field][indices_npy] = pc[field][map_indices_npy]

    cdef _update_gradients(self, CarrayContainer pc, CarrayContainer gradient, list fields):
        """
        Transfer gradient from image particle to ghost particle.

        Parameters
        ----------
        pc : CarrayContainer
            Gradient data
        fields : list
            List of field strings to update
        """
        cdef int i, j, ip, dim = self.domain.dim
        cdef np.float64_t *x[3], *dv[4]

        cdef LongArray indices = LongArray()
        cdef IntArray types = pc.get_carray("type")
        cdef np.ndarray indices_npy, map_indices_npy

        # find all ghost that need to be updated
        for i in range(pc.get_number_of_items()):
            if types.data[i] == Exterior:
                indices.append(i)

        indices_npy = indices.get_npy_array()
        map_indices_npy = pc["map"][indices_npy]

        # update ghost with their image data
        for field in fields:
            gradient[field][indices_npy] = gradient[field][map_indices_npy]

        # refective bc, mirror velocity gradients
        if self.boundary_type == BoundaryType.Reflective:

            pc.pointer_groups(x, pc.named_groups['position'])
            gradient.pointer_groups(dv, gradient.named_groups['velocity'])

            for i in range(indices_npy.size): # every exterior ghost
                for j in range(dim): # each dimension

                    ip = indices.data[i]

                    # reverse the normal of velocity gradient
                    if x[j][ip] < self.domain.bounds[0][j]:
                        # flip gradient component
                        dv[(dim+1)*j][ip] *= -1

                    if x[j][ip] > self.domain.bounds[1][j]:
                        # flip gradient component
                        dv[(dim+1)*j][ip] *= -1

