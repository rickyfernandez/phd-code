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
        Add needed fields for interaction computation to particle
        container. Note if fields already present this function
        does nothing.

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
                pc.register_carray(num_particles, field,
                        self.fields[field])

        # add needed groups as well
        for group in self.carray_named_groups.iterkeys():
            if group in pc.carray_named_groups:
                for field in group:
                    if field not in pc.carray_named_groups[group]:
                        pc.carray_named_groups[group] = list(self.carray_named_groups[group])
                        break
            else:
                pc.carray_named_groups[group] = list(self.carray_named_groups[group])

    cdef void particle_not_finished(self, long node_index):
        """
        Flag current particle as not finished in walk and save next
        node index to restart walk.

        Parameters
        ---------
        node_index : long
            index of node to restart tree walk
        """
        self.particle_done = False
        self.current_node = node_index

    cdef bint done_processing(self):
        """
        Signal if all particles have partcipatd in walk

        Returns
        -------
        bint
            Boolean to signal if all particles have been walked
        """
        if self.current < self.num_particles:
            return False
        else:
            return True

    cdef void particle_finished(self):
        """
        Flag current particle as finished in walk
        """
        self.particle_done = True

    cdef long start_node_index(self):
        """
        Return the starting node index for walk. If particle is starting walk
        root is returned otherwise next node is returned to restart walk.

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

    cdef void initialize_particles(self, CarrayContainer pc, bint local_particles=True):
        msg = "InteractionBase::initialize_particles called!"
        raise NotImplementedError(msg)

    cdef bint process_particle(self):
        msg = "InteractionBase::process_particle called!"
        raise NotImplementedError(msg)

cdef class GravityAcceleration(Interaction):
    """
    Compute acceleration due to gravity between particle and
    gravity node
    """
    def __init__(self, CarrayContainer pc, DomainLimits domain, Splitter splitter,
            int add_fields=0, int calculate_potential=0, double smoothing_length=0.0):
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
        self.smoothing_length = smoothing_length
        self.calc_potential = calculate_potential

        self.fields = {}
        self.carray_named_groups = {}

        self.fields['mass'] = 'double'
        self.carray_named_groups['position'] = []
        self.carray_named_groups['acceleration'] = []

        for axis in 'xyz'[:self.dim]:
            self.fields['position-' + axis] = 'double'
            self.fields['acceleration-' + axis] = 'double'
            self.carray_named_groups['position'].append('position-' + axis)
            self.carray_named_groups['acceleration'].append('acceleration-' + axis)

        self.carray_named_groups['gravity'] = ['mass'] + self.carray_named_groups['position'] +\
                self.carray_named_groups['acceleration']
        self.carray_named_groups['gravity-walk-export'] = ['mass'] + pc.carray_named_groups['position']
        self.carray_named_groups['gravity-walk-import'] = list(self.carray_named_groups['acceleration'])

        if self.calc_potential:
            self.fields['potential'] = 'double'
            self.carray_named_groups['gravity-walk-import'] = self.carray_named_groups['gravity-walk-import']\
                    + ['potential']

        # if splitter has fields add as well 
        self.splitter.add_fields_to_interaction(self.fields, self.carray_named_groups)

        # add new fields to container
        if add_fields:
            self.add_fields_to_container(pc)

    cdef void initialize_particles(self, CarrayContainer pc, bint local_particles=True):
        """
        Set referecne to particles for walk

        Parameters
        ----------
        pc : CarrayContainer
            Container of particles that are going to walk the tree
        local_particles : bint
            Flag indicating if we are using particles local to our processor
        """
        cdef DoubleArray doub_array

        self.num_particles = pc.get_number_of_items()

        self.current = -1
        self.particle_done = 1
        self.current_node = ROOT
        self.local_particles = local_particles

        # set reference for acceleration calculation
        if self.local_particles:
            self.tags = pc.get_carray('tag')
        pc.pointer_groups(self.a, pc.carray_named_groups['acceleration'])
        pc.pointer_groups(self.x, pc.carray_named_groups['position'])

        # reference for potential calculation
        if self.calc_potential:
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
        # happens only on local processor
        if(node.flags & LEAF):
            if(self.local_particles):
                if(node.group.data.pid == self.current):
                    return

        r2 = 0.
        # distance between particle and center of mass
        for i in range(self.dim):
            dr[i] = node.group.data.com[i] - self.x[i][self.current]
            r2 += dr[i]**2

        # add smoothing length
        r2 += self.smoothing_length**2

        # particle acceleration
        fac = node.group.data.mass / (sqrt(r2) * r2)
        for i in range(self.dim):
            self.a[i][self.current] += fac * dr[i]

        # particle potential per mass
        if self.calc_potential:
            self.pot[self.current] -= node.group.data.mass / sqrt(r2)

    cdef bint process_particle(self):
        """
        Iterator for particles. Each time this is called it moves
        on to the next particle, skipping ghost particles, for calculation.
        Assumes initialize_particles was called before processing particles.

        Returns
        -------
        bint
           True if all particels have walked otherwise False
        """
        # continue to next particle if done walking previous particle
        if self.particle_done:
            self.current += 1

            # local particels can have ghost so need to skip in
            # gravity calculation
            if self.local_particles:
                if self.current < self.num_particles:

                    # skip ghost particles
                    while(self.tags[self.current] == Ghost):
                        if self.current + 1 < self.num_particles:
                            self.current += 1
                        else:
                            # all particles walked
                            return False

            if self.current < self.num_particles:
                # setup particle for walk
                for i in range(self.dim):
                    self.a[i][self.current] = 0.
                if self.calc_potential:
                    self.pot[self.current] = 0.

                # set splitter to next particle
                self.splitter.process_particle(self.current)
                # particle ready for walk
                return True
            else:
                # all particles walked
                return False
        else:
            # particle ready to restart walk
            return True
