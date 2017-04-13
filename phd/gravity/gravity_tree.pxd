cimport numpy as np
from ..utils.carray cimport IntArray
from ..domain.domain cimport DomainLimits
from ..load_balance.tree cimport Node as LoadNode
from ..containers.containers cimport CarrayContainer
from ..load_balance.load_balance cimport LoadBalance

# --- add later to reduce memory overhead ---
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

    double x[3]        # particle position
    double mass        # particle mass

cdef struct Node:

    Particle p

    int leaf           # flag if node is a leaf
    double width       # physical width of node
    int dependant      # if node depends on nodes from another processor
    double center[3]   # physical center of the node

    int children[8]    # array of indicies of children in node array

    # for efficient tree walking
    int first_child
    int next_sibling

cdef class GravityNodePool:

    cdef int used                     # number of nodes used in the pool
    cdef int capacity                 # total capacity of the pool

    cdef Node* node_array             # array holding all nodes

    cdef Node* get(self, int count)   # allocate count many nodes
    cdef void resize(self, int size)  # resize array of nodes to length size
    cdef void reset(self)             # reset the pool
    cpdef int number_leaves(self)     # number of leves in tree
    cpdef int number_nodes(self)      # number of nodes in tree

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

    cdef public int dim, rank, size
    cdef public int number_nodes
    cdef public DomainLimits domain
    cdef public GravityNodePool nodes

    # varaibles for parallel run
    cdef int parallel
    cdef public LoadBalance load_bal
    cdef public CarrayContainer remote_nodes
    cdef public np.ndarray node_counts
    cdef public np.ndarray node_disp

    # allocate node functions
    cdef inline int get_index(self, int parent_index, Particle* p)
    cdef inline Node* create_child(self, int parent_index, int child_index)
    cdef inline void create_children(self, int parent_index)

    # tree build functions
#    #cdef void _build_tree(self, CarrayContainer pc)
    cdef void _build_top_tree(self)
    cdef void _create_top_tree(self, int node_index, LoadNode* load_parent,
            np.int32_t* node_map)
    cdef int _leaf_index_toptree(self, np.int64_t key)

    # moment calculation functions
    cdef void _update_moments(self, int current, int sibling)
    cdef void _export_import_remote_nodes(self)
    cdef void _update_remote_moments(self, int current)
#
#    # tree walk functions
#    #cdef void _walk(self, Interaction interaction, CarrayContainer pc)
