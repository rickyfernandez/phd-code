import phd


cdef inline bint in_box(double x[3], double r, np.float64_t bounds[2][3], int dim):
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

cdef class BoundaryConditionBase:
    cdef void create_ghost_particle(self, FlagParticle *p, DomainManager domain_manager):
        if phd._in_parallel:
            self.create_ghost_particle_parallel(p, domain_manager)
        else:
            self.create_ghost_particle_serial(p, domain_manager)

    cdef void create_ghost_particle_serial(self, FlagParticle *p, DomainManager domain_manager):
        msg = "BoundaryBase::create_ghost_particle_serial called!"
        raise NotImplementedError(msg)

    cdef void create_ghost_particle_parallel(self, FlagParticle *p, DomainManager domain_manager):
        msg = "BoundaryBase::create_ghost_particle_serial!"
        raise NotImplementedError(msg)

    cdef void migrate_particles(self, CarrayContainer particles, DomainManager domain_manager):
        msg = "BoundaryBase::create_ghost_particle_serial!"
        raise NotImplementedError(msg)

cdef class Reflective(BoundaryConditionBase):
    cdef void create_ghost_particle_serial(self, FlagParticle *p, DomainManager domain_manager):
        """
        Create reflective ghost particles in the simulation. Can be used in non-parallel
        and parallel runs. Ghost particles are appended right after real particles in
        the container. Particle container should only have real particles when used.

        Parameters
        ----------
        """
        cdef int i, k
        cdef double xs[3], vs[3]
        cdef int dim = domain_manager.domain.dim

        # skip particle does not intersect boundary
        if in_box(p.x, p.search_radius, domain_manager.domain.bounds, dim):

            # extract real particle
            for i in range(dim):

                # lower boundary
                # does particle radius leave global boundary 
                # skip if processed in earlier iteration 
                if p.x[i] < domain_manager.domain.translate[i]/2.0:
                    if(p.x[i] - p.search_radius < domain_manager.domain.bounds[0][i]) and\
                        (p.x[i] - p.old_search_radius > domain_manager.domain.bounds[0][i]):

                        # copy particle information
                        for k in range(dim):
                            xs[k] = p.x[k]
                            vs[k] = p.v[k]

                        # reflect particle position and velocity 
                        xs[i] =  xs[i] - 2*(xs[i] - domain_manager.domain.bounds[0][i])
                        vs[i] = -vs[i]

                        # create ghost particle
                        domain_manager.ghost_vec.push_back(
                                BoundaryParticle(xs, vs,
                                    p.index, 0, REFLECTIVE, dim))

                # upper boundary
                # does particle radius leave global boundary 
                # skip if processed in earlier iteration 
                if p.x[i] > domain_manager.domain.translate[i]/2.0:
                    if (domain_manager.domain.bounds[1][i] < p.x[i] + p.search_radius) and\
                        (domain_manager.domain.bounds[1][i] > p.x[i] + p.old_search_radius):

                        # copy particle information
                        for k in range(dim):
                            xs[k] = p.x[k]
                            vs[k] = p.v[k]

                        # reflect particle position and velocity
                        xs[i] =  xs[i] - 2*(xs[i] - domain_manager.domain.bounds[1][i])
                        vs[i] = -vs[i]

                        # create ghost particle
                        domain_manager.ghost_vec.push_back(
                                BoundaryParticle(xs, vs,
                                    p.index, 0, REFLECTIVE, dim))

    cdef void migrate_particles(self, CarrayContainer particles, DomainManager domain_manager):
        pass
#
#        cdef np.float64_t xp[3], *x[3]
#        cdef int i, j, dim, is_outside
#
#        particles.remove_tagged_particles(ParticleTAGS.Ghost)
#
#        dim = len(particles.named_groups['position'])
#        particles.pointer_groups(x, particles.named_groups['position'])
#
#        for i in range(particles.get_number_of_items()):
#
#            # did particle leave domain
#            is_outside = 0
#            for j in range(dim):
#                xp[j] = x[j][i]
#                is_outside += xp[j] <= domain_manager.domain.bounds[0][j] or domain_manager.domain.bounds[1][j] <= xp[j]
#
#            if is_outside: # particle left domain
#                raise RuntimeError("particle left domain in reflective boundary condition!!")


#    cdef void create_ghost_particle_serial(self, np.float64_t xp[3], DomainManager domain_manager):
#        cdef int k
#        cdef int dim = domain_manager.domain.dim
#
#        for k in range(dim):
#            if not in_box(p.x, 0.0, domain_manager.domain.bounds, dim):
#                raise RuntimeError("particle left domain in reflective boundary condition!!")

cdef class Periodic(BoundaryConditionBase):
    cdef void create_ghost_particle_serial(self, FlagParticle *p, DomainManager domain_manager):
        """
        Create periodic ghost particles in the simulation. Should only be used in
        serial run.
        """
        cdef int j, k
        cdef double xs[3]
        cdef int index[3]
        cdef int dim = domain_manager.domain.dim

        # skip partilce if does not intersect boundary
        if in_box(p.x, p.search_radius, domain_manager.domain.bounds, dim):

            # shift particle
            for i in range(3**dim):

                # create shift indices
                index[0] = i%3; index[1] = (i/3)%3; index[2] = i/9
                if (i == 4 and dim == 2) or (i == 13 and dim == 3):
                    continue # skip no shift

                # shifted particle
                for k in range(dim):
                    xs[k] = p.x[k] +\
                            (index[k]-1)*domain_manager.domain.translate[k]

                # find if shifted particle intersects domain
                # skip if processed in earlier iteration 
                if(in_box(xs, p.search_radius,
                        domain_manager.domain.bounds, dim)) and\
                                not (in_box(xs, p.old_search_radius,
                                    domain_manager.domain.bounds, dim)):

                        # create ghost particle
                        domain_manager.ghost_vec.push_back(
                                BoundaryParticle(xs, p.v,
                                    p.index, 0, PERIODIC, dim))

#    cdef void create_ghost_particle_serial(self, np.float64_t x[3], np.float64_t *xp[3], DomainManager domain_manager):
#        cdef int k
#        cdef int dim = domain_manager.domain.dim
#
#        for j in range(dim):
#            if x[j] <= self.domain.bounds[0][j]:
#                xp[j][i] += self.domain.translate[j]
#            if x[j] >= self.domain.bounds[1][j]:
#                xp[j][i] -= self.domain.translate[j]

#cdef class Reflective(BoundaryBase):
#    cdef void create_ghost_particle_parallel(self, FlagParticle *p, DomainManager domain_manager)
#        """
#        Create reflective ghost particles in the simulation. Can be used in non-parallel
#        and parallel runs. Ghost particles are appended right after real particles in
#        the container. Particle container should only have real particles when used.
#
#        Parameters
#        ----------
#        """
#        cdef int i, k
#        cdef double xs[3], vs[3]
#        cdef dim = domain_manager.dim
#
#        # skip if it does not intersect boundary
#        if in_box(p.x, p.search_radius, domain.bounds, dim):
#
#            # extract real particle
#            for i in range(dim):
#
#                # lower boundary
#                # does particle radius leave global boundary 
#                if p.x[i] - p.search_radius < domain_manager.domain.bounds[0][i]:
#
#                    # copy particle information
#                    for k in range(dim):
#                        xp[k] = p.x[k]
#                        vp[k] = p.v[k]
#
#                    # reflect particle position and velocity 
#                    xp[i] =  xp[i] - 2*(xp[i] - domain_manager.domain.bounds[0][i])
#                    vp[i] = -vp[i]
#
#                    # find processor neighbors
#                    domain_manager.processor_intersection(
#                            xs, p.radius, p.search_radius)
#
#                    if domain_manger.processor_nbrs.size():
#                        for i in range(domain_manager.processor_nbrs.size()):
#                            domain_manager.ghost_vec.push_back(
#                                    BoundaryParticle(
#                                        xs, p.v,
#                                        p.index,
#                                        domain_manager.processor_nbrs[i],
#                                        REFLECTIVE,
#                                        dim))
#
#                # upper boundary
#                # does particle radius leave global boundary 
#                if domain_manager.domain.bounds[1][j] < p.x[j] + p.search_radius:
#
#                    # copy particle information
#                    for k in range(dim):
#                        xp[k] = p.x[k]
#                        vp[k] = p.v[k]
#
#                    # reflect particle position and velocity
#                    xp[j] =  xp[j] - 2*(xp[j] - domain_manager.domain.bounds[1][j])
#                    vp[j] = -vp[j]
#
#                    # find processor neighbors
#                    domain_manager.processor_intersection(
#                            xs, p.radius, p.search_radius)
#
#                    # copy particle for every flagged processor
#                    if domain_manger.processor_nbrs.size():
#                        for i in range(domain_manager.processor_nbrs.size()):
#                            domain_manager.ghost_vec.push_back(
#                                    BoundaryParticle(
#                                        xs, p.v,
#                                        p.index,
#                                        domain_manager.processor_nbrs[i],
#                                        REFLECTIVE,
#                                        dim))
#
#class Periodic(BoundaryBase):
#    cdef void create_ghost_particle_parallel(self, FlagParticle *p, DomainManager domain_manager)
#        """
#        Create periodic ghost particles in the simulation. Should only be used in
#        """
#        cdef int i, j, k
#        cdef double xs[3]
#        cdef int index[3]
#        cdef int dim = domain_manager.dim
#
#        # skip if it does not intersect boundary
#        if in_box(p.x, p.search_radius, domain.bounds, dim):
#
#            # shift particle
#            for i in range(3**dim):
#
#                # create shift indices
#                index[0] = i%3; index[1] = (i/3)%3; index[2] = i/9
#                if (i == 4 and dim == 2) or (i == 13 and dim == 3):
#                    continue # skip no shift
#
#                # copy particle
#                for k in range(dim):
#                    xs[k] = p.x[k]
#
#                # shifted position
#                for k in range(dim):
#                    xs[k] = p.x[k] +\
#                            (index[k]-1)*domain_manager.domain.translate[k]
#
#                # find processor neighbors
#                domain_manager.processor_intersection(
#                        xs, p.radius, p.search_radius)
#
#                # copy particle for every flagged process 
#                if domain_manger.processor_nbrs.size():
#                    for j in range(domain_manager.processor_nbrs.size()):
#                        domain_manager.ghost_particle.push_back(
#                                BoundaryParticle(
#                                    xs, p.v,
#                                    p.index,
#                                    domain_manager.processor_nbrs[j],
#                                    PERIODIC,
#                                    dim))
#
