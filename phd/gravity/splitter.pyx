
cdef class Splitter:
    cdef void initialize_particles(self, CarrayContainer pc):
        msg = "Splitter::initialize_particles called!"
        raise NotImplementedError(msg)

    cdef void process_particle(self, long idp):
        self.idp = idp

    cdef int split(self, Node *node):
        msg = "Splitter::split called!"
        raise NotImplementedError(msg)

cdef class BarnesHut(Splitter):
    def __init__(self, double open_angle):
        self.open_angle = open_angle

    cdef void initialize_particles(self, CarrayContainer pc):
        pc.pointer_groups(self.x, pc.named_groups['position'])

    cdef int split(self, Node* node):
        """
        Test if node needs to be open using the Barnes and Hut Criteria.
        """
        cdef int i
        cdef double r2 = 0.

        for i in range(self.dim):
            r2 += (self.x[i][self.idp] - node.group.data.com[i])**2

        if(node.width*node.width >= r2*self.open_angle*self.open_angle):
            return 1
        else:
            return 0
