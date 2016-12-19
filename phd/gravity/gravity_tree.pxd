cimport numpy as np
from ..utils.carray cimport IntArray
from ..domain.domain cimport DomainLimits
from ..containers.containers cimport CarrayContainer


#cdef struct Moments:
#    double mass
#    double com[3]
#    double cmax
#    double vmax
#    double hmax

#cdef union Info:
#    int children[8]
#    struct Moments mom

cdef struct Particle:

    double x[3]
    double mass

cdef struct Node:

    int leaf
    Particle p
    int children[8]

    double center[3]
    double width

#    Info info

    # for efficient tree walking
    int first_child
    int next_sibling

cdef class GravityNodePool:

    cdef int used                         # number of nodes used in the pool
    cdef int capacity                     # total capacity of the pool

    cdef Node* node_array                 # array holding all nodes

    cdef Node* get(self, int count) nogil       # return 'count' many nodes
    cdef void resize(self, int size) nogil      # resize array of nodes to length size
    cdef void reset(self)                 # reset the pool
    cpdef int number_leaves(self)         # number of leves in tree
    cpdef int number_nodes(self)          # number of nodes in tree

cdef class Splitter:

    cdef int dim
    cdef long idp

    cdef void initialize_particles(self, CarrayContainer pc)
    cdef void process_particle(self, long idp)
    cdef int split(self, Node* node)

cdef class BarnesHut(Splitter):
    cdef double open_angle
    cdef np.float64_t *x[3]

cdef class Interaction:

    cdef int dim
    cdef long current
    cdef long num_particles

    cdef IntArray tags
    cdef Splitter splitter

    cdef void interact(self, Node* node)
    cdef void initialize_particles(self, CarrayContainer pc)
    cdef int process_particle(self)
    cpdef void set_splitter(self, Splitter splitter)

cdef class GravityAcceleration(Interaction):
    cdef np.float64_t *x[3]
    cdef np.float64_t *a[3]

cdef class GravityTree:

    cdef int dim
    cdef int number_nodes
    cdef DomainLimits domain

    cdef Node* root
    cdef public GravityNodePool nodes

    cdef void _build_tree(self, CarrayContainer pc)
    cdef inline int get_index(self, Node* node, Particle* p) nogil
    cdef inline Node* create_child(self, Node* parent, int index) nogil
    cdef void _update_moments(self, int current, int sibling) nogil
    cdef void _walk(self, Interaction interaction, CarrayContainer pc)
