from libc.math cimport sqrt

from .gravity_tree cimport Node, LEAF, ROOT
from ..utils.carray cimport DoubleArray
from ..domain.domain cimport DomainLimits
from ..utils.particle_tags import ParticleTAGS
from ..load_balance.tree cimport TreeMemoryPool as Pool


cdef int Ghost = ParticleTAGS.Ghost

cdef class Interaction:
    """
    Base class for particle node computation
    """
    def __init__(self, CarrayContainer pc, DomainLimits domain, Splitter splitter, int add_fields=0):
        self.dim = domain.dim
        self.splitter = splitter

    def add_fields_to_container(self, CarrayContainer pc):
        """
        Add needed fields for interaction computation
        to particle container. Note if fields present this
        function does nothing.

        Parameters
        ----------
        pc : CarrayContainer
            Container holding particle data
        """
        cdef str field, group
        cdef long num_particles

        num_particles = pc.get_number_of_items()

        # add needed fields for computation
        for field in self.fields.iterkeys():
            if field not in pc.properties:
                pc.register_property(num_particles, field,
                        self.fields[field])

        # add needed groups as well
        for group in self.named_groups.iterkeys():
            if group in pc.named_groups:
                for field in group:
                    if field not in pc.named_groups[group]:
                        pc.named_groups[group] = list(self.named_groups[group])
                        break
            else:
                pc.named_groups[group] = list(self.named_groups[group])

    cdef void particle_not_finished(self, long node_index):
        """
        Flag current particle as not finished in walk and save next
        node index in walk for restarting walk.

        Parameters
        ---------
        node_index : long
            index of node to restart tree walk
        """
        self.particle_done = 0
        self.current_node = node_index

    cdef int done_processing(self):
        """
        """
        if self.current < self.num_particles:
            return 0
        else:
            return 1

    cdef int particle_finished(self):
        """
        """
        self.particle_done = 1

    cdef long start_node_index(self):
        """
        Return the starting node index for walk. If particle is starting walk
        then root is returned else the next node is returned to restart walk.

        Returns
        -------
        long
            Index of node to start walk
        """
        if self.particle_done:
            return ROOT
        else:
            return self.current_node

    cdef void interact(self, Node* node):
        msg = "InteractionBase::interact called!"
        raise NotImplementedError(msg)

    cdef void initialize_particles(self, CarrayContainer pc, int has_ghost=1):
        msg = "InteractionBase::initialize_particles called!"
        raise NotImplementedError(msg)

    cdef int process_particle(self):
        msg = "InteractionBase::process_particle called!"
        raise NotImplementedError(msg)

cdef class GravityAcceleration(Interaction):
    """
    Compute acceleration due to gravity between particle and
    gravity node
    """
    def __init__(self, CarrayContainer pc, DomainLimits domain, Splitter splitter,
            int add_fields=0, int calculate_potential=0):
        """
        Initialize gravity interaction

        Parameters
        ----------

        domain : DomainLimits
            Domain of simulation
        potential : int
            Flag to calculate potential
        """
        cdef str axis

        self.dim = domain.dim
        self.splitter = splitter
        self.calc_potential = calculate_potential

        self.fields = {}
        self.named_groups = {}

        self.fields['mass'] = 'double'
        self.named_groups['position'] = []
        self.named_groups['acceleration'] = []

        for axis in 'xyz'[:self.dim]:
            self.fields['position-' + axis] = 'double'
            self.fields['acceleration-' + axis] = 'double'
            self.named_groups['position'].append('position-' + axis)
            self.named_groups['acceleration'].append('acceleration-' + axis)

        self.named_groups['gravity'] = ['mass'] + self.named_groups['position'] +\
                self.named_groups['acceleration']
        self.named_groups['gravity-walk-export'] = ['mass'] + pc.named_groups['position']
        self.named_groups['gravity-walk-import'] = list(self.named_groups['acceleration'])

        if self.calc_potential:
            self.fields['potential'] = 'double'
            self.named_groups['gravity-walk-import'] = self.named_groups['gravity-walk-import']\
                    + ['potential']

        # if splitter has fields add as well 
        self.splitter.add_fields_to_interaction(self.fields, self.named_groups)

        # add new fields to container
        if add_fields:
            self.add_fields_to_container(pc)

    cdef void initialize_particles(self, CarrayContainer pc, int has_ghost=1):
        """
        Set referecne to particles for walk

        Parameters
        ----------
        pc : CarrayContainer
            Container of particles that are going to walk the tree
        """
        cdef DoubleArray doub_array

        self.has_ghost = has_ghost
        self.num_particles = pc.get_number_of_items()

        self.current = -1
        self.particle_done = 1
        self.current_node = ROOT
        self.has_ghost = has_ghost

        if self.has_ghost:
            self.tags = pc.get_carray('tag')
        pc.pointer_groups(self.a, pc.named_groups['acceleration'])
        pc.pointer_groups(self.x, pc.named_groups['position'])

        if self.calc_potential:
            doub_array = pc.get_carray('mass')
            self.m     = doub_array.get_data_ptr()
            doub_array = pc.get_carray('potential')
            self.pot   = doub_array.get_data_ptr()

        # setup information for opening nodes and
        # first particle to process
        self.splitter.initialize_particles(pc)

    cdef void interact(self, Node* node):
        """
        Calculate acceleration of particle due to node

        Parameters
        ----------
        node : *Node
            Gravity node from gravity tree
        """
        cdef int i, inside
        cdef double fac, r2, dr[3]

        # ignore self interaction
        if(node.flags & LEAF):
            if(self.has_ghost): # need better flag because doubles for remote
                if(node.group.data.pid == self.current):
                    return

        r2 = 0.
        for i in range(self.dim):
            # distance between particle and center of mass
            dr[i] = node.group.data.com[i] - self.x[i][self.current]
            r2 += dr[i]**2

        # particle acceleration
        fac = node.group.data.mass / (sqrt(r2) * r2)
        for i in range(self.dim):
            self.a[i][self.current] += fac * dr[i]

        # particle potential
        if self.calc_potential:
            fac = self.m[self.current] * node.group.data.mass / sqrt(r2)
            for i in range(self.dim):
                self.pot[self.current] += fac

    cdef int process_particle(self):
        """
        Iterator for particles. Each time this is called it moves
        on to the next particle, skipping ghost particles, for calculation.
        Assumes initialize_particles was called before processing particles.

        Returns
        -------
        int
            If done processing all real particles returns 0 otherwise 1
        """
        # continue to next particle if done walking previous particle
        if self.particle_done:
            self.current += 1

            if self.has_ghost:
                if self.current < self.num_particles:
                    # skip ghost particles
                    while(self.tags[self.current] == Ghost):
                        if self.current + 1 < self.num_particles:
                            self.current += 1
                        else:
                            return 0

            if self.current < self.num_particles:
                # setup particle for walk
                for i in range(self.dim):
                    self.a[i][self.current] = 0.
                if self.calc_potential:
                    self.pot[self.current] = 0.

                self.splitter.process_particle(self.current)
                return 1
            else:
                return 0

        else:
            return 1
