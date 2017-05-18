cdef class Splitter:
    """
    Base class for open node criteria.
    """
    cdef void initialize_particles(self, CarrayContainer pc):
        msg = "Splitter::initialize_particles called!"
        raise NotImplementedError(msg)

    cdef void process_particle(self, long idp):
        self.idp = idp

    cdef int split(self, Node *node):
        msg = "Splitter::split called!"
        raise NotImplementedError(msg)

cdef class BarnesHut(Splitter):
    """
    Barnes and Hut criteria to open node.
    """
    def __init__(self, int dim, double open_angle):
        """
        Initialize class with opening angle

        Parameters
        ----------
        open_angle : double
            opening angle criteria
        """
        self.dim = dim
        self.open_angle = open_angle

    def add_fields_to_interaction(self, dict fields, dict named_groups):
        pass

    cdef void initialize_particles(self, CarrayContainer pc):
        """
        Create reference to particle positions

        Parameters
        ----------
        pc : CarrayContainer
            Container of particles that are going to walk the tree
        """
        pc.pointer_groups(self.x, pc.named_groups['position'])

    cdef int split(self, Node* node):
        """
        Test if node needs to be open using the Barnes and Hut Criteria.

        Parameters
        ----------
        node : *Node
            Node in gravity tree to test
        """
        cdef int i
        cdef double r2 = 0.

        for i in range(self.dim):
            r2 += (self.x[i][self.idp] - node.group.data.com[i])**2

        if(node.width*node.width >= r2*self.open_angle*self.open_angle):
            return 1
        else:
            return 0
