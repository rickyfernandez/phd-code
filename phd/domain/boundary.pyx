import phd

from ..utils.carray cimport IntArray, LongArray
from ..utils.particle_tags import ParticleTAGS

cdef int REAL = ParticleTAGS.Real
cdef int EXTERIOR = ParticleTAGS.Exterior

cdef inline bint intersect_bounds(double x[3], double r, np.float64_t bounds[2][3], int dim):
    """Check if box with center x and half edge r intersects with
    bounds.

    Parameters
    ----------
    x : array[3]
        Box center position.

    r : np.float64_t
        Half edge of box.

    bounds : array[2][3]
        Bounds of interest, min/max in each dimension.

    dim : int
        Problem dimension.

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
        msg = "BoundaryBase::create_ghost_particle_serial called!"
        raise NotImplementedError(msg)

    cdef void migrate_particles(self, CarrayContainer particles, DomainManager domain_manager):
        msg = "BoundaryBase::create_ghost_particle_serial called!"
        raise NotImplementedError(msg)

    cdef void update_gradients(self, CarrayContainer particles, CarrayContainer gradients,
                               DomainManager domain_manager):
        msg = "BoundaryBase::update_gradients called!"
        raise NotImplementedError(msg)

cdef class Reflective(BoundaryConditionBase):
    cdef void create_ghost_particle_serial(self, FlagParticle *p, DomainManager domain_manager):
        """Create reflective ghost particles in serial run.

        Parameters
        ----------
        p : FlagParticle*
            Pointer to flagged particle.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        cdef int i, k
        cdef double xs[3], vs[3]
        cdef int dim = domain_manager.domain.dim
        cdef double pos_new, pos_old, domaind_edge

        # skip particle does not intersect boundary
        if intersect_bounds(p.x, p.search_radius, domain_manager.domain.bounds, dim):
            for i in range(dim):

                pos_new = p.x[i] - p.search_radius 
                pos_old = p.x[i] - p.old_search_radius 
                domain_edge = domain_manager.domain.bounds[0][i]

                # lower boundary
                # does particle radius leave global boundary 
                # skip if processed in earlier iteration 
                if p.x[i] <= domain_manager.domain.translate[i]/2.0:
                    if (pos_new <= domain_edge) and (pos_old > domain_edge):

                        # copy particle information
                        for k in range(dim):
                            xs[k] = p.x[k]
                            vs[k] = p.v[k]

                        # reflect particle position and velocity 
                        xs[i] =  xs[i] - 2*(xs[i] - domain_edge)
                        vs[i] = -vs[i]

                        # create ghost particle
                        domain_manager.ghost_vec.push_back(
                                BoundaryParticle(xs, vs,
                                    p.index, 0, dim))

                pos_new = p.x[i] + p.search_radius 
                pos_old = p.x[i] + p.old_search_radius 
                domain_edge = domain_manager.domain.bounds[1][i]

                # upper boundary
                # does particle radius leave global boundary 
                # skip if processed in earlier iteration 
                if p.x[i] > domain_manager.domain.translate[i]/2.0:
                    if (pos_new >= domain_edge) and (pos_old < domain_edge):

                        # copy particle information
                        for k in range(dim):
                            xs[k] = p.x[k]
                            vs[k] = p.v[k]

                        # reflect particle position and velocity
                        xs[i] =  xs[i] - 2*(xs[i] - domain_edge)
                        vs[i] = -vs[i]

                        # create ghost particle
                        domain_manager.ghost_vec.push_back(
                                BoundaryParticle(xs, vs,
                                    p.index, 0, dim))

#    cdef void create_ghost_particle_parallel(self, FlagParticle *p, DomainManager domain_manager)
#        """Create reflective ghost particles in parallel run.
#
#        Parameters
#        ----------
#        p : FlagParticle*
#            Pointer to flagged particle.
#
#        domain_manager : DomainManager
#            Class that handels all things related with the domain.
#
#        """
#        cdef int i, j, k
#        cdef double xs[3], vs[3]
#        cdef dim = domain_manager.dim
#        cdef double pos_new, domaind_edge
#        cdef LongArray proc_nbrs = LongArray()
#
#        # skip if it does not intersect boundary
#        if intersect_bounds(p.x, p.search_radius, domain.bounds, dim):
#            for i in range(dim):
#
#                pos_new = p.x[i] - p.search_radius 
#                domain_edge = domain_manager.domain.bounds[0][i]
#
#                # lower boundary
#                # does particle radius leave global boundary 
#                # skip if processed in earlier iteration 
#                if p.x[i] <= domain_manager.domain.translate[i]/2.0:
#                    if (pos_new <= domain_edge):
#
#                        # copy particle information
#                        for k in range(dim):
#                            xs[k] = p.x[k]
#                            vs[k] = p.v[k]
#
#                        # reflect particle position and velocity 
#                        xs[i] =  xs[i] - 2*(xs[i] - domain_edge)
#                        vs[i] = -vs[i]
#
#                        # find processor neighbors, proc=-1 is used to
#                        # query our own processor
#                        proc_nbrs.reset()
#                        domain_manager.processor_neighbors_intersection(
#                                xs, p.old_search_radius, p.search_radius,
#                                proc_nbrs, -1)
#
#                        if proc_nbrs.size():
#                            for j in range(proc_nbrs.size()):
#                                domain_manager.ghost_vec.push_back(
#                                        BoundaryParticle(
#                                            xs, p.v, p.index,
#                                            proc_nbrs[j], dim))
#
#                pos_new = p.x[i] + p.search_radius 
#                domain_edge = domain_manager.domain.bounds[1][i]
#
#                # upper boundary
#                # does particle radius leave global boundary 
#                # skip if processed in earlier iteration 
#                if p.x[i] > domain_manager.domain.translate[i]/2.0:
#                    if (pos_new >= domain_edge):
#
#                        # copy particle information
#                        for k in range(dim):
#                            xs[k] = p.x[k]
#                            vs[k] = p.v[k]
#
#                        # reflect particle position and velocity
#                        xs[i] =  xs[i] - 2*(xs[i] - domain_edge)
#                        vs[i] = -vs[i]
#
#                        # find processor neighbors, proc=-1 is used to
#                        # query our own processor
#                        proc_nbrs.reset()
#                        domain_manager.processor_neighbors_intersection(
#                                xs, p.old_search_radius, p.search_radius,
#                                proc_nbrs, -1)
#
#                        if proc_nbrs.size():
#                            for j in range(proc_nbrs.size()):
#                                domain_manager.ghost_vec.push_back(
#                                        BoundaryParticle(
#                                            xs, p.v, p.index,
#                                            proc_nbrs[j], dim))

#    cdef void migrate_particles(self, CarrayContainer particles, DomainManager domain_manager):
#        """After a time step particles are moved. This implements the boundary
#        condition to particles that have left the domain.
#
#        Parameters
#        ----------
#        particles : CarrayContainer
#            Class that holds all information pertaining to the particles.
#
#        domain_manager : DomainManager
#            Class that handels all things related with the domain.
#
#        """
#        cdef int k, dim
#        cdef double xs[3]
#        cdef np.float64_t *x[3]
#        cdef IntArray tags = particles.get_carray("tag")
#
#        dim = len(particles.carray_named_groups["position"])
#        particles.pointer_groups(x, particles.carray_named_groups["position"])
#
#        for i in range(particles.get_carray_size()):
#            if tags.data[i] == REAL:
#                for k in range(dim):
#                    xs[k] = x[k][i]
#
#                if not intersect_bounds(xs, 0.0, domain_manager.domain.bounds, dim):
#                    raise RuntimeError("particle left domain in reflective boundary condition!!")


#    cdef void update_gradients(self, CarrayContainer particles, CarrayContainer gradients,
#                               DomainManager domain_manager):
#        """Transfer gradient from image particle to ghost particle with reflective
#        boundary condition.
#
#        For reflective boundary condition the velocity gradient normal to the
#        boundary interface has to be flipped. This function finds all ghost
#        particles outside the domain and flips the corresponding component.
#
#        Parameters
#        ----------
#        particles : CarrayContainer
#            Gradient data
#
#        gradients : CarrayContainer
#            Container of gradients for each primitive field.
#
#        """
#        cdef str field
#        cdef int i, j, k, ip
#        cdef np.float64_t *x[3], *dv[4]
#
#        cdef LongArray indices = LongArray()
#        cdef IntArray types = particles.get_carray("type")
#        cdef np.ndarray indices_npy, map_indices_npy
#
#        dim = len(particles.carray_named_groups["position"])
#
#        # find all ghost that need to be updated
#        for i in range(particles.get_carray_size()):
#            if types.data[i] == EXTERIOR:
#                indices.append(i)
#
#        # each ghost particle knows the id from which
#        # it was created from
#        indices_npy = indices.get_npy_array()
#        map_indices_npy = particles["map"][indices_npy]
#
#        # update ghost with their image data
#        for field in gradients.carray_named_groups["primitive"]:
#            gradients[field][indices_npy] = gradients[field][map_indices_npy]
#
#        particles.pointer_groups(x, particles.carray_named_groups["position"])
#        gradients.pointer_groups(dv, gradients.carray_named_groups["velocity"])
#
#        for i in range(indices_npy.size): # every exterior ghost
#
#            # extract particle
#            ip = indices.data[i]
#
#            # check each dimension
#            for j in range(dim):
#
#                if x[j][ip] < domain_manager.domain.bounds[0][j]:
#                    for k in range(dim):
#                        # flip gradient component
#                        dv[dim*k + j][ip] *= -1
#
#                if x[j][ip] > domain_manager.domain.bounds[1][j]:
#                    for k in range(dim):
#                        # flip gradient component
#                        dv[dim*k + j][ip] *= -1
#

cdef class Periodic(BoundaryConditionBase):
    cdef void create_ghost_particle_serial(self, FlagParticle *p, DomainManager domain_manager):
        """Create periodic ghost particles in serial run.

        Parameters
        ----------
        p : FlagParticle*
            Pointer to flagged particle.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        cdef int j, k
        cdef double xs[3]
        cdef int index[3]
        cdef int dim = domain_manager.domain.dim

        # skip partilce if does not intersect boundary
        if intersect_bounds(p.x, p.search_radius, domain_manager.domain.bounds, dim):

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
                if intersect_bounds(xs, p.search_radius, domain_manager.domain.bounds, dim) and\
                        not intersect_bounds(xs, p.old_search_radius, domain_manager.domain.bounds, dim):

                    # create ghost particle
                    domain_manager.ghost_vec.push_back(
                            BoundaryParticle(xs, p.v,
                                p.index, 0, dim))

    cdef void create_ghost_particle_parallel(self, FlagParticle *p, DomainManager domain_manager):
        """Create periodic ghost particles in parallel run.

        Parameters
        ----------
        p : FlagParticle*
            Pointer to flagged particle.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        cdef int i, j, k
        cdef double xs[3]
        cdef int index[3]
        cdef int dim = domain_manager.dim
        cdef LongArray proc_nbrs = LongArray()

        # skip if it does not intersect boundary
        if intersect_bounds(p.x, p.search_radius, domain_manager.domain.bounds, dim):

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

                # find processor neighbors
                proc_nbrs.reset()
                domain_manager.processor_neighbors_intersection(
                        xs, p.old_search_radius, p.search_radius,
                        proc_nbrs, phd._rank)

                if proc_nbrs.size():
                    for j in range(proc_nbrs.size()):
                        domain_manager.ghost_vec.push_back(
                                BoundaryParticle(
                                    xs, p.v, p.index,
                                    proc_nbrs[j], dim))

#    cdef void update_gradients(self, CarrayContainer particles, CarrayContainer gradients,
#                               DomainManager domain_manager):
#        """Transfer gradient from image particle to ghost particle with reflective
#        boundary condition.
#
#        For periodic boundary condition the there is no need to update gradients.
#
#        Parameters
#        ----------
#        particles : CarrayContainer
#            Gradient data
#
#        gradients : CarrayContainer
#            Container of gradients for each primitive field.
#        p : FlagParticle*
#            Pointer to flagged particle.
#
#        domain_manager : DomainManager
#            Class that handels all things related with the domain.
#
#        """
#        pass

#    cdef void migrate_particles(self, CarrayContainer particles, DomainManager domain_manager):
#        """After a time step particles are moved. This implements the periodic boundary
#        condition to particles that have left the domain.
#
#        Parameters
#        ----------
#        particles : CarrayContainer
#            Class that holds all information pertaining to the particles.
#
#        domain_manager : DomainManager
#            Class that handels all things related with the domain.
#
#        """
#        cdef int k, dim
#        cdef double xs[3]
#        cdef np.float64_t *x[3]
#
#        cdef IntArray tags = particles.get_carray("tag")
#
#        dim = len(particles.carray_named_groups["position"])
#        particles.pointer_groups(x, particles.carray_named_groups["position"])
#
#        for i in range(particles.get_carray_size()):
#            if tags.data[i] == REAL:
#                for k in range(dim):
#                    xs[k] = x[k][i]
#
#                if not intersect_bounds(xs, 0.0, domain_manager.domain.bounds, dim):
#
#                    for k in range(dim):
#                        if xs[k] <= domain_mananger.domain.bounds[0][j]:
#                            x[k][i] += domain_manager.translate[j]
#                        if xp[k] >= domain_manager.domain.bounds[1][j]:
#                            x[k][i] -= domain_manager.translate[j]
#
