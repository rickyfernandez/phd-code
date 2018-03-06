import phd
import numpy as np
from libc.math cimport sqrt

from .gravity_tree cimport Node, LEAF, ROOT, TOP_TREE_LEAF
from ..utils.carray cimport DoubleArray
from ..utils.particle_tags import ParticleTAGS
from ..load_balance.tree cimport TreeMemoryPool as Pool


cdef int GHOST = ParticleTAGS.Ghost


cdef class Interaction:
    """Base class for particle node computation."""

    cdef void particle_not_finished(self, long node_index):
        """Flag current particle as not finished in walk and save
        next node index to restart walk.

        Parameters
        ---------
        node_index : long
            Index of node to restart tree walk.

        """
        self.particle_done = False
        self.current_node = node_index

    cdef bint done_processing(self):
        """Signal if all particles have partcipatd in walk.

        Returns
        -------
        bint
            Boolean to signal if all particles have been walked.

        """
        if self.current < self.num_particles:
            return False
        else:
            return True

    cdef void particle_finished(self):
        """Flag current particle as finished in walk."""
        self.particle_done = True

    cdef long start_node_index(self):
        """Return the starting node index for walk. If particle is
        starting walk root is returned otherwise last node is
        returned to continue walk.

        Returns
        -------
        long
            Index of node to start walk.

        """
        if self.particle_done:
            return ROOT
        else:
            return self.current_node

    cdef void interact(self, Node* node):
        msg = "InteractionBase::interact called!"
        raise NotImplementedError(msg)

    cdef void initialize_particles(self, CarrayContainer particles,
                                   bint local_particles=True):
        msg = "InteractionBase::initialize_particles called!"
        raise NotImplementedError(msg)

    cdef bint process_particle(self):
        msg = "InteractionBase::process_particle called!"
        raise NotImplementedError(msg)

cdef class GravityAcceleration(Interaction):
    """Class to compute acceleration due to gravity between particle
    and gravity node.
    """
    def __init__(self, int calculate_potential=0,
                 double smoothing_length=0.0):
        """Initialize interaction class.

        Parameters
        ----------
        potential : int
            Flag if potential calculation is needed.

        smoothing_length : int
            Gravitational smoothing length.

        """
        self.splitter = None
        self.particle_fields_registered = False
        self.smoothing_length = smoothing_length
        self.calculate_potential = calculate_potential

        if phd._in_parallel:
            self.flag_pid = np.zeros(phd._size, dtype=np.int32)

    def set_splitter(self, Splitter splitter):
        """Set criteria to open node."""
        self.splitter = splitter

    def register_fields(self, CarrayContainer particles):
        """Add needed fields for interaction computation to particle
        container.

        Parameters
        ----------
        particles : CarrayContainer
            Container holding particle data.

        """
        cdef str axis
        cdef int num_particles

        dim = len(particles.carray_named_groups["position"])
        num_particles = particles.get_carray_size()

        particles.carray_named_groups["acceleration"] = []

        # add acceleration fields
        for axis in "xyz"[:dim]:
            particles.register_carray(num_particles,
                    "acceleration-"+axis, "double")
            particles.carray_named_groups["acceleration"].append(
                    "acceleration-"+axis)

        if self.calculate_potential:
            particles.register_carray(num_particles,
                    "potential", "double")

        # gravity group
        particles.carray_named_groups["gravity"] = ["mass"] +\
                particles.carray_named_groups["position"] +\
                particles.carray_named_groups["acceleration"]

        if phd._in_parallel:

            # we export mass and position for tree walk on other processors 
            particles.carray_named_groups["gravity-walk-export"] = ["mass"] +\
                    particles.carray_named_groups["position"]

            # we only import acceleration of tree walk on onther processors
            particles.carray_named_groups["gravity-walk-import"] =\
                    list(particles.carray_named_groups["acceleration"])

            if self.calculate_potential:
                particles.carray_named_groups["gravity-walk-import"] +=\
                        ["potential"]

        self.dim = dim
        self.particle_fields_registered = True

    def initialize(self):
        """Initialze class making sure all initial routines
        have been called.
        """
        if not self.particle_fields_registered:
            raise RuntimeError("ERROR:Fields not registered in\
                                particles by Splitter!")

        if not self.splitter:
            raise RuntimeError("ERROR:Splitter not defined")

    cdef void initialize_particles(self, CarrayContainer particles,
                                   bint local_particles=True):
        """Setup parameters for tree walk with  reference to particle
        acceleration and position.

        Parameters
        ----------
        particles : CarrayContainer
            Container of particles that are going to walk the tree.

        local_particles : bint
            Flag indicating if we are using particles local to our
            processor.

        """
        cdef DoubleArray doub_array

        self.num_particles = particles.get_carray_size()

        self.current = -1
        self.particle_done = True
        self.current_node = ROOT
        self.local_particles = local_particles

        # set reference for acceleration calculation
        if self.local_particles:
            self.tags = particles.get_carray("tag")

        particles.pointer_groups(self.a,
                particles.carray_named_groups["acceleration"])
        particles.pointer_groups(self.x,
                particles.carray_named_groups["position"])

        # reference for potential calculation
        if self.calculate_potential:
            doub_array = particles.get_carray("potential")
            self.pot   = doub_array.get_data_ptr()

        # setup information for opening nodes and
        # first particle to process
        self.splitter.initialize_particles(particles)

    cdef void interact(self, Node* node):
        """Calculate acceleration of particle due to node.

        Parameters
        ----------
        node : *Node
            Gravity node from gravity tree.

        """
        cdef int i, inside
        cdef double fac, r2, dr[3]

        # ignore self interaction
        # happens only on local processor
        if(self.local_particles):
            if(node.flags & LEAF):
                if((node.flags & TOP_TREE_LEAF) != TOP_TREE_LEAF):
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
        if self.calculate_potential:
            self.pot[self.current] -= node.group.data.mass / sqrt(r2)

    cdef bint process_particle(self):
        """Iterator for particles. Each time this is called it moves
        to the next particle, skipping ghost particles, unless previous
        particle calculation not finished. Assumes initialize_particles
        was called before processing particles.

        Returns
        -------
        bint
           True if all particles have walked otherwise False.

        """
        # continue to next particle if done walking previous particle
        if self.particle_done:
            self.current += 1

            # local particels can have ghost so need to skip in
            # gravity calculation
            if self.local_particles:
                if self.current < self.num_particles:

                    # skip ghost particles
                    while(self.tags[self.current] == GHOST):
                        if self.current + 1 < self.num_particles:
                            self.current += 1
                        else:
                            # all particles walked
                            return False

            if self.current < self.num_particles:

                # clear out acceleration for walk
                for i in range(self.dim):
                    self.a[i][self.current] = 0.
                if self.calculate_potential:
                    self.pot[self.current] = 0.

                if phd._in_parallel:
                    # clear our export flag
                    for i in range(phd._size):
                        self.flag_pid[i] = 0

                # set splitter to next particle
                self.splitter.process_particle(self.current)
                # particle ready for walk
                return True
            else:
                # all particles walked
                return False
        else:
            # particle ready to restart walk
            self.splitter.process_particle(self.current)
            return True
