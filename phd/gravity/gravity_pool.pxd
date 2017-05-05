cdef struct Data:

    double mass         # mass of node
    double com[3]       # center of mass of node
    int first_child     # first child of node
    int next_sibling    # next sibling of node
    int pid             # particle index

cdef union Group:

    int children[8]    # reference to nodes children
    Data data          # Data struct

cdef struct Node:

    int flags          # flags
    double width       # physical width of node
    double center[3]   # physical center of the node
    Group group        # union of moment information and children index

cdef class GravityPool:

    cdef int used                     # number of nodes used in the pool
    cdef int capacity                 # total capacity of the pool

    cdef Node* array                  # array holding all nodes

    cdef Node* get(self, int count)   # allocate count many nodes
    cdef void resize(self, int size)  # resize array of nodes to length size
    cdef void reset(self)             # reset the pool
    cpdef int number_leaves(self)     # number of leves in tree
    cpdef int number_nodes(self)      # number of nodes in tree
