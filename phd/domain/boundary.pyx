import phd

from libcpp.list cimport list as cpplist
from cython.operator cimport preincrement as inc

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
    cdef void create_ghost_particle(self, cpplist[FlagParticle] &flagged_particles,
                                    DomainManager domain_manager):
        if phd._in_parallel:
            self.create_ghost_particle_parallel(flagged_particles, domain_manager)
        else:
            self.create_ghost_particle_serial(flagged_particles, domain_manager)

    cdef void create_ghost_particle_serial(self, cpplist[FlagParticle] &flagged_particles,
                                           DomainManager domain_manager):
        msg = "BoundaryBase::create_ghost_particle_serial called!"
        raise NotImplementedError(msg)

    cdef void create_ghost_particle_parallel(self, cpplist[FlagParticle] &flagged_particles,
                                             DomainManager domain_manager):
        msg = "BoundaryBase::create_ghost_particle_serial called!"
        raise NotImplementedError(msg)

    cdef void migrate_particles(self, CarrayContainer particles, DomainManager domain_manager):
        msg = "BoundaryBase::create_ghost_particle_serial called!"
        raise NotImplementedError(msg)

    cdef void update_gradients(self, CarrayContainer particles, CarrayContainer gradients,
                               DomainManager domain_manager):
        msg = "BoundaryBase::update_gradients called!"
        raise NotImplementedError(msg)

    cpdef update_fields(self, CarrayContainer particles, DomainManager domain_manager):
        msg = "BoundaryBase::update_fields called!"
        raise NotImplementedError(msg)

cdef class Reflective(BoundaryConditionBase):
    cdef void create_ghost_particle_serial(self, cpplist[FlagParticle] &flagged_particles,
                                           DomainManager domain_manager):
        """Create reflective ghost particles in serial run.

        Parameters
        ----------
        p : FlagParticle*
            Pointer to flagged particle.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        cdef int i, k
        cdef double xs[3]
        cdef FlagParticle *p
        cdef int dim = domain_manager.domain.dim
        cdef double pos_new, pos_old, domaind_edge

        cdef cpplist[FlagParticle].iterator it = flagged_particles.begin()
        while it != flagged_particles.end():
            p = particle_flag_deref(it)

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

                            # reflect particle position and velocity 
                            xs[i] =  xs[i] - 2*(xs[i] - domain_edge)

                            # create ghost particle
                            domain_manager.ghost_vec.push_back(
                                    BoundaryParticle(xs, p.index,
                                        0, EXTERIOR, dim))

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

                            # reflect particle position and velocity
                            xs[i] =  xs[i] - 2*(xs[i] - domain_edge)

                            # create ghost particle
                            domain_manager.ghost_vec.push_back(
                                    BoundaryParticle(xs, p.index,
                                        0, EXTERIOR, dim))

            inc(it)  # increment iterator

    cdef void create_ghost_particle_parallel(self, cpplist[FlagParticle] &flagged_particles,
                                             DomainManager domain_manager):
        """Create reflective ghost particles in parallel run.

        Parameters
        ----------
        p : FlagParticle*
            Pointer to flagged particle.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        cdef int i, j, k
        cdef double xs[3]
        cdef FlagParticle *p
        cdef double pos_new, domain_edge
        cdef int dim = domain_manager.domain.dim
        cdef LongArray proc_nbrs = LongArray()

        cdef cpplist[FlagParticle].iterator it = flagged_particles.begin()
        while it != flagged_particles.end():
            p = particle_flag_deref(it)

            # skip if it does not intersect boundary
            if intersect_bounds(p.x, p.search_radius, domain_manager.domain.bounds, dim):
                for i in range(dim):

                    pos_new = p.x[i] - p.search_radius
                    domain_edge = domain_manager.domain.bounds[0][i]

                    # lower boundary
                    # does particle radius leave global boundary 
                    # skip if processed in earlier iteration 
                    if p.x[i] <= domain_manager.domain.translate[i]/2.0:
                        if (pos_new <= domain_edge):

                            # copy particle information
                            for k in range(dim):
                                xs[k] = p.x[k]

                            # reflect particle position and velocity 
                            xs[i] =  xs[i] - 2*(xs[i] - domain_edge)

                            # find processor neighbors, proc=-1 is used to
                            # query our own processor
                            proc_nbrs.reset()
                            domain_manager.get_nearest_intersect_process_neighbors(
                                    xs, p.old_search_radius, p.search_radius,
                                    -1, proc_nbrs)

                            if proc_nbrs.length:
                                for j in range(proc_nbrs.length):
                                    domain_manager.ghost_vec.push_back(
                                            BoundaryParticle(
                                                xs, p.index,
                                                proc_nbrs.data[j],
                                                EXTERIOR, dim))

                    pos_new = p.x[i] + p.search_radius
                    domain_edge = domain_manager.domain.bounds[1][i]

                    # upper boundary
                    # does particle radius leave global boundary 
                    # skip if processed in earlier iteration 
                    if p.x[i] > domain_manager.domain.translate[i]/2.0:
                        if (pos_new >= domain_edge):

                            # copy particle information
                            for k in range(dim):
                                xs[k] = p.x[k]

                            # reflect particle position and velocity
                            xs[i] =  xs[i] - 2*(xs[i] - domain_edge)

                            # find processor neighbors, proc=-1 is used to
                            # query our own processor
                            proc_nbrs.reset()
                            domain_manager.get_nearest_intersect_process_neighbors(
                                    xs, p.old_search_radius, p.search_radius,
                                    -1, proc_nbrs)

                            if proc_nbrs.length:
                                for j in range(proc_nbrs.length):
                                    domain_manager.ghost_vec.push_back(
                                            BoundaryParticle(
                                                xs, p.index,
                                                proc_nbrs.data[j],
                                                EXTERIOR, dim))

            inc(it)  # increment iterator

    cdef void migrate_particles(self, CarrayContainer particles, DomainManager domain_manager):
        """After a time step particles are moved. This implements the boundary
        condition to particles that have left the domain.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        cdef int k, dim
        cdef double xs[3]
        cdef np.float64_t *x[3]
        cdef IntArray tags = particles.get_carray("tag")

        dim = len(particles.carray_named_groups["position"])
        particles.pointer_groups(x, particles.carray_named_groups["position"])

        for i in range(particles.get_carray_size()):
            if tags.data[i] == REAL:
                for k in range(dim):
                    xs[k] = x[k][i]

                if not intersect_bounds(xs, 0.0, domain_manager.domain.bounds, dim):
                    raise RuntimeError("particle left domain in reflective boundary condition!!")

    cdef void update_gradients(self, CarrayContainer particles, CarrayContainer gradients,
                               DomainManager domain_manager):
        """Transfer gradient from image particle to ghost particle with reflective
        boundary condition.

        For reflective boundary condition the velocity gradient normal to the
        boundary interface has to be flipped. This function finds all ghost
        particles outside the domain and flips the corresponding component.

        Parameters
        ----------
        particles : CarrayContainer
            Gradient data

        gradients : CarrayContainer
            Container of gradients for each primitive field.

        """
        cdef int i, j, k, dim
        cdef np.float64_t *x[3], *dv[9]
        cdef IntArray types = particles.get_carray("type")

        dim = len(particles.carray_named_groups["position"])
        particles.pointer_groups(x, particles.carray_named_groups["position"])
        gradients.pointer_groups(dv, gradients.carray_named_groups["velocity"])

        for i in range(particles.get_carray_size()):
            if types.data[i] == EXTERIOR:

                # check each dimension
                for j in range(dim):

                    if x[j][i] < domain_manager.domain.bounds[0][j]:
                        for k in range(dim):
                            # flip gradient component
                            dv[dim*k + j][i] *= -1

                    if x[j][i] > domain_manager.domain.bounds[1][j]:
                        for k in range(dim):
                            # flip gradient component
                            dv[dim*k + j][i] *= -1

    cpdef update_fields(self, CarrayContainer particles, DomainManager domain_manager):
        """Transfer gradient from image particle to ghost particle with reflective
        boundary condition.

        For reflective boundary condition the velocity gradient normal to the
        boundary interface has to be flipped. This function finds all ghost
        particles outside the domain and flips the corresponding component.

        Parameters
        ----------
        particles : CarrayContainer
            Gradient data

        gradients : CarrayContainer
            Container of gradients for each primitive field.

        """
        cdef int i, k, dim
        cdef np.float64_t *x[3], *v[3], *mv[3]
        cdef IntArray types = particles.get_carray("type")

        dim = len(particles.carray_named_groups["position"])

        particles.pointer_groups(x,  particles.carray_named_groups["position"])
        particles.pointer_groups(v,  particles.carray_named_groups["velocity"])
        particles.pointer_groups(mv, particles.carray_named_groups["momentum"])

        for i in range(particles.get_carray_size()):
            if types.data[i] == EXTERIOR:

                # check each dimension
                for k in range(dim):

                    if x[k][i] < domain_manager.domain.bounds[0][k]:
                        v[k][i]  *= -1
                        mv[k][i] *= -1

                    if x[k][i] > domain_manager.domain.bounds[1][k]:
                        v[k][i]  *= -1
                        mv[k][i] *= -1

cdef class Periodic(BoundaryConditionBase):
    cdef void create_ghost_particle_serial(self, cpplist[FlagParticle] &flagged_particles,
                                           DomainManager domain_manager):
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
        cdef FlagParticle *p
        cdef int dim = domain_manager.domain.dim

        cdef cpplist[FlagParticle].iterator it = flagged_particles.begin()
        while it != flagged_particles.end():
            p = particle_flag_deref(it)

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
                                BoundaryParticle(xs, p.index,
                                    0, EXTERIOR, dim))

            inc(it)  # increment iterator

    cdef void create_ghost_particle_parallel(self, cpplist[FlagParticle] &flagged_particles,
                                             DomainManager domain_manager):
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
        cdef LongArray proc_nbrs = LongArray()
        cdef int dim = domain_manager.domain.dim

        cdef cpplist[FlagParticle].iterator it = flagged_particles.begin()
        while it != flagged_particles.end():
            p = particle_flag_deref(it)

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
                    domain_manager.get_nearest_intersect_process_neighbors(
                            xs, p.old_search_radius, p.search_radius,
                            phd._rank, proc_nbrs)

                    if proc_nbrs.length:
                        for j in range(proc_nbrs.length):
                            domain_manager.ghost_vec.push_back(
                                    BoundaryParticle(
                                        xs, p.index,
                                        proc_nbrs.data[j],
                                        EXTERIOR, dim))

            inc(it)  # increment iterator

    cdef void update_gradients(self, CarrayContainer particles, CarrayContainer gradients,
                               DomainManager domain_manager):
        """Transfer gradient from image particle to ghost particle with reflective
        boundary condition.

        For periodic boundary condition the there is no need to update gradients.

        Parameters
        ----------
        particles : CarrayContainer
            Gradient data

        gradients : CarrayContainer
            Container of gradients for each primitive field.
        p : FlagParticle*
            Pointer to flagged particle.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        pass

    cpdef update_fields(self, CarrayContainer particles, DomainManager domain_manager):
        """Transfer gradient from image particle to ghost particle with reflective
        boundary condition.

        For periodic boundary condition the there is no need to update gradients.

        Parameters
        ----------
        particles : CarrayContainer
            Gradient data

        gradients : CarrayContainer
            Container of gradients for each primitive field.
        p : FlagParticle*
            Pointer to flagged particle.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        pass

    cdef void migrate_particles(self, CarrayContainer particles, DomainManager domain_manager):
        """After a time step particles are moved. This implements the periodic boundary
        condition to particles that have left the domain.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        cdef double xs[3]
        cdef int i, k, dim
        cdef np.float64_t *x[3]
        cdef IntArray tags = particles.get_carray("tag")

        dim = len(particles.carray_named_groups["position"])
        particles.pointer_groups(x, particles.carray_named_groups["position"])

        for i in range(particles.get_carray_size()):
            if tags.data[i] == REAL:
                for k in range(dim):
                    xs[k] = x[k][i]

                if not intersect_bounds(xs, 0.0, domain_manager.domain.bounds, dim):
                    # wrap particle back into domain
                    for k in range(dim):

                        # lower boundary
                        if xs[k] <= domain_manager.domain.bounds[0][k]:
                            x[k][i] += domain_manager.domain.translate[k]

                        # upper boundary
                        if xs[k] >= domain_manager.domain.bounds[1][k]:
                            x[k][i] -= domain_manager.domain.translate[k]
