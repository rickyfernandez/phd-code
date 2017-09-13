
def difference(LongArray A, LongArray B, LongArray C):
    cdef int i, j
    cdef int asize = A.size
    cdef int bsize = B.size

    C.resize(0)

    i = j = 0
    while(True):

        if i == asize:
            while(j < bsize):
                C.append(B[j])
                j += 1
            return

        if j == bsize:
            while(i < asize):
                C.append(A[i])
                i += 1
            return

        if A[i] < B[j]:
            C.append(A[i])
            i += 1
        elif A[i] > B[j]:
            C.append(B[j])
            j += 1
        else:
            i += 1
            j += 1

cdef class DomainManager:
    def __init__(self):

        self.domain = None
        self.load_balance = None
        self.boundary_condition = None

        # search radius of particles
        self.old_particles_radius = DoubleArray()

        # interior particles 
        self.old_interior_flagged_particles = LongArray()
        self.new_interior_flagged_particles = LongArray()

        # exterior particles 
        self.old_exterior_flagged_particles = LongArray()
        self.new_exterior_flagged_particles = LongArray()

    @check_class(phd.Domain)
    def set_domain(domain):
        '''add boundary condition to list'''
        self.domain = domain

    @check_class(phd.BoundaryConditionBase)
    def set_boundary_condition(boundary_condition):
        '''add boundary condition to list'''
        self.boundary_condition = boundary_condition

    @check_class(phd.LoadBalance)
    def set_boundary_condition(boundary_condition):
        '''add boundary condition to list'''
        self.boundary_condition = boundary_condition

    cpdef partition(self, CarrayContainer particles):
        """
        Distribute particles across processors.
        """
        pass

    cdef set_initial_radius(self, CarrayContainer particles):
        """
        Starting radius for each particle in the simulation.

        Parameters
        ----------
        pc : CarrayContainer
            Particle data
        """
        cdef int i
        cdef double fac = self.scale_factor
        cdef Tree glb_tree = self.load_bal.tree
        cdef double box_size = self.domain.max_length
        cdef DoubleArray r = particles.get_carray("radius")

        for i in range(particles.get_number_of_items()):
            if phd._in_parallel:
                node = glb_tree.find_leaf(keys.data[i])
                r.data[i] = fac*node.box_length/glb_tree.domain_fac
            else:
                r.data[i] = fac*box_size


    cdef create_ghost_particles(CarrayContainer particles, bint first_attempt):

        # go through old interior particles and find which need to
        # be sent to processor that have been sent to yet

        num_flagged_interior = self.new_interior_flagged_particles.size()
        num_flagged_exterior = self.new_exterior_flagged_particles.size()

        # resize for copying
        self.old_interior_flagged_particles.resize(num_flagged_interior)
        self.old_exterior_flagged_particles.resize(num_flagged_exterior)

        # copy particls that where previously flagged
        for i in range(num_flagged_interior):
            self.old_interior_flagged_particles[i] =\
                    self.new_interior_flagged_particles[i]
        self.new_interior_flagged_particles.clear()

        for i in range(num_flagged_exterior):
            self.old_exterior_flagged_particles[i] =\
                    self.new_exterior_flagged_particles[i]
        self.new_exterior_flagged_particles.clear()

        # create new exterior ghost particles
        for j in range(num_flagged_exterior):

            # extract particle
            i = self.old_flagged_particles[j]
            for k in range(dim):
                p.x[k] = x[k][i]
                p.v[k] = v[k][i]

            p.old_radius = self.radius.data[j]
            p.new_radius = radius.data[i]
            p.index = i

            if self.boundary.create_ghost_particle(
                    p, domain_mangager):
                self.new_exterior_flagged_particles.append(i)
                self.new_exterior_flagged_particles.append(i)

        self.new_interior_flagged_particles.clear()
        for j in range(num_flagged_interior):

            # extract particle
            i = self.old_flagged_particles[j]
            for k in range(dim):
                p.x[k] = x[k][i]
                p.v[k] = v[k][i]

            p.old_radius = self.radius.data[j]
            p.new_radius = radius.data[i]
            p.index = i

            if self.create_interior_ghost_particle(
                    p, domain_mangager):
                self.new_interior_flagged_particles.append(i)
                self.new_interior_flagged_particles.append(i)

        # copy particles, put in processor order and export
        self.copy_particles(particles, ghost_particles, ghost_vec)

    cdef copy_particles(CarrayContainer particles, CarrayContainer ghost_particles):
        """
        Copy particles from ghost_particle vector
        """

        cdef CarrayContainer exterior_ghost
        cdef DoubleArray mass

        cdef int i, j
        cdef Particle *p
        cdef np.float64_t *x[3], *v[3], *mv[3]
        cdef np.float64_t *xg[3], *vg[3], *mv[3]

        # create ghost particles from flagged particles
        exterior_ghost = particles.extract_items(indices)

        # references to new ghost data
        exterior_ghost.pointer_groups(xg, particles.named_groups['position'])
        exterior_ghost.pointer_groups(vg, particles.named_groups['velocity'])

        mass = exterior_ghost.get_carray("mass")
        exterior_ghost.pointer_groups(mv, particles.named_groups['momentum'])

        # transfer new data to ghost 
        for i in range(exterior_ghost.get_number_of_items()):

            maps.data[i] = indices.data[i]  # reference to image

            # update new position/velocity
            p = &ghost_particle[i]
            for j in range(dim):

                xg[j][i] = p.x[j]
                vg[j][i] = p.v[j]
                mv[j][i] = mass.data[i]*p.v[j]

        # add new ghost to particle container
        ghost_particles.append_container(exterior_ghost)

    cdef bint ghost_no_complete(self):
        return (self.new_exterior_flagged_particles.empty() and\
                self.new_interior_flagged_particles.empty())
