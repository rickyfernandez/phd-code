import phd


cdef bint inline in_box(np.float64_t x[3], np.float64_t r,
        np.float64_t bounds[2][3], int dim):
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
    cdef create_ghost_particle(QueryParticle *p, DomainManager domain_manager):
        if phd._in_parallel:
            self.create_ghost_particle_parallel(p, domain_manager)
        else:
            self.create_ghost_particle_serial(p, domain_manager)

    cdef bint create_ghost_particle_serial(QueryParticle *p, DomainManager domain_manager):
        msg = "BoundaryBase::create_ghost_particle_serial called!"
        raise NotImplementedError(msg)

    cdef bint create_ghost_particle_parallel(QueryParticle *p, DomainManager domain_manager):
        msg = "BoundaryBase::create_ghost_particle_serial!"
        raise NotImplementedError(msg)

cdef class Reflective(BoundaryBase):
    cdef bint create_ghost_particle_serial(QueryParticle *p, DomainManager domain_manager)
        """
        Create reflective ghost particles in the simulation. Can be used in non-parallel
        and parallel runs. Ghost particles are appended right after real particles in
        the container. Particle container should only have real particles when used.

        Parameters
        ----------
        """
        cdef int i, k
        cdef double xs[3], vs[3]
        cdef int dim = domain_manager.dim
        cdef bint particle_flagged = False

        # extract real particle
        for i in range(dim):

            # lower boundary
            # does particle radius leave global boundary 
            if p.x[i] - p.new_radius < domain_manager.domain.bounds[0][i]:
                particle_flagged = True

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
                            p.index, 0, dim))

            # upper boundary
            # does particle radius leave global boundary 
            if domain_manager.domain.bounds[1][i] < p.x[i] + p.new_radius:
                particle_flagged = True

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
                            p.index, 0, dim))

        return particle_flagged

#    cdef bint create_ghost_particle_parallel(QueryParticle *p, DomainManager domain_manager)
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
#        cdef bint particle_flagged = False
#
#        # extract real particle
#        for i in range(dim):
#
#            # lower boundary
#            # does particle radius leave global boundary 
#            if p.x[i] - p.new_radius < domain_manager.domain.bounds[0][i]:
#
#                # copy particle information
#                for k in range(dim):
#                    xp[k] = p.x[k]
#                    vp[k] = p.v[k]
#
#                # reflect particle position and velocity 
#                xp[i] =  xp[i] - 2*(xp[i] - domain_manager.domain.bounds[0][i])
#                vp[i] = -vp[i]
#
#                # find processor neighbors
#                domain_manager.processor_intersection(
#                        xs, radius1, radius2)
#
#                if domain_manger.processor_nbrs.size():
#                    for i in range(domain_manager.processor_nbrs.size()):
#                        domain_manager.ghost_vec.push_back(
#                                BoundaryParticle(
#                                    xs, p.v,
#                                    p.index,
#                                    domain_manager.processor_nbrs[i],
#                                    dim))
#
#            # upper boundary
#            # does particle radius leave global boundary 
#            if domain_manager.domain.bounds[1][j] < p.x[j] + p.new_radius:
#
#                # copy particle information
#                for k in range(dim):
#                    xp[k] = p.x[k]
#                    vp[k] = p.v[k]
#
#                # reflect particle position and velocity
#                xp[j] =  xp[j] - 2*(xp[j] - domain_manager.domain.bounds[1][j])
#                vp[j] = -vp[j]
#
#                # find processor neighbors
#                domain_manager.processor_intersection(
#                        xs, radius1, radius2)
#
#                # copy particle for every flagged process 
#                if domain_manger.processor_nbrs.size():
#                    particle_flagged = True # particle not finished
#                    for i in range(domain_manager.processor_nbrs.size()):
#                        domain_manager.ghost_vec.push_back(
#                                BoundaryParticle(
#                                    xs, p.v,
#                                    p.index,
#                                    domain_manager.processor_nbrs[i],
#                                    dim))
#
#        return particle_flagged

class Periodic(BoundaryBase):
    cdef bint create_ghost_particle_serial(QueryParticle *p, DomainManager domain_manager)
        """
        Create periodic ghost particles in the simulation. Should only be used in
        """
        cdef int j, k
        cdef double xs[3]
        cdef int index[3]
        cdef int dim = domain_manager.dim
        cdef bint particle_flagged = False

        # shift particle
        for i in range(3**dim):

            # create shift indices
            index[0] = i%3; index[1] = (i/3)%3; index[2] = i/9
            if (i == 4 and dim == 2) or (i == 13 and dim == 3):
                continue # skip no shift

            # shifted position
            for k in range(dim):
                xs[k] = p.x[k] +\
                        (index[k]-1)*domain_manager.domain.translate[k]

            # find if shifted particle intersects domain
            if in_box(xs, p.new_radius,
                    domain_manager.domain.bounds, dim):

                # create ghost particle
                domain_manager.ghost_vec.push_back(
                        BoundaryParticle(xs, p.vp,
                            p.index, 0, dim))

        return particle_flagged

#    cdef bint create_ghost_particle_parallel(QueryParticle *p, DomainManager domain_manager)
#        """
#        Create periodic ghost particles in the simulation. Should only be used in
#        """
#        cdef int i, j, k
#        cdef double xs[3]
#        cdef int index[3]
#        cdef int dim = domain_manager.dim
#        cdef bint particle_flagged = False
#
#        # shift particle
#        for i in range(3**dim):
#
#            # create shift indices
#            index[0] = i%3; index[1] = (i/3)%3; index[2] = i/9
#            if (i == 4 and dim == 2) or (i == 13 and dim == 3):
#                continue # skip no shift
#
#            # copy particle
#            for k in range(dim):
#                xs[k] = p.x[k]
#
#            # shifted position
#            for k in range(dim):
#                xs[k] = p.x[k] +\
#                        (index[k]-1)*domain_manager.domain.translate[k]
#
#            # find processor neighbors
#            domain_manager.processor_intersection(
#                    xs, p.radius1, p.radius2)
#
#            # copy particle for every flagged process 
#            if domain_manger.processor_nbrs.size():
#                particle_flagged = True # particle not finished
#                for j in range(domain_manager.processor_nbrs.size()):
#                    domain_manager.ghost_particle.push_back(
#                            BoundaryParticle(
#                                xs, p.v,
#                                p.index,
#                                domain_manager.processor_nbrs[j],
#                                dim))
#
#        return particle_flagged
