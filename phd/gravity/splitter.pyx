cdef class Splitter:
    """Base class for open node criteria."""
    def set_dim(self, int dim):
        """Set dimension of simulation."""
        self.dim = dim

    cdef void initialize_particles(self, CarrayContainer particles):
        msg = "Splitter::initialize_particles called!"
        raise NotImplementedError(msg)

    cdef void process_particle(self, long idp):
        self.idp = idp

    cdef int split(self, Node *node):
        msg = "Splitter::split called!"
        raise NotImplementedError(msg)

cdef class BarnesHut(Splitter):
    """Barnes and Hut criteria."""
    def __init__(self, double open_angle):
        """Initialize class with opening angle.

        Parameters
        ----------
        open_angle : double
            opening angle criteria.

        """
        self.open_angle = open_angle

    cdef void initialize_particles(self, CarrayContainer particles):
        """Create reference to particle positions.

        Parameters
        ----------
        particles : CarrayContainer
            Container of particles that are going to walk the tree.

        """
        particles.pointer_groups(self.x, particles.carray_named_groups["position"])

    cdef int split(self, Node* node):
        """Test if node needs to be open using the Barnes and Hut Criteria.

        Parameters
        ----------
        node : *Node
            Node in gravity tree to test.

        Returns
        -------
        int
            If 1 open node otherwise 0.

        """
        cdef int i
        cdef double r2 = 0.

        for i in range(self.dim):
            r2 += (self.x[i][self.idp] - node.group.data.com[i])**2

        if(node.width*node.width >= r2*self.open_angle*self.open_angle):
            return 1 # open node
        else:
            return 0 # dont open node
