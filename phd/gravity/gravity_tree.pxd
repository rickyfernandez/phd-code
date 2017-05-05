cimport numpy as np

from .gravity_pool cimport Node, GravityPool

from ..domain.domain cimport DomainLimits
from ..load_balance.tree cimport Node as LoadNode
from ..containers.containers cimport CarrayContainer
from ..load_balance.load_balance cimport LoadBalance

# tree flags
cdef int NOT_EXIST = -1
cdef int ROOT = 0
cdef int ROOT_SIBLING = -1
cdef int LEAF = 0x01
cdef int HAS_PARTICLE = 0x02
cdef int TOP_TREE = 0x04
cdef int TOP_TREE_LEAF = 0x08
cdef int TOP_TREE_LEAF_REMOTE = 0x10
cdef int SKIP_BRANCH = 0x20


cdef class GravityTree:

    cdef public int number_nodes
    cdef public int dim, rank, size
    cdef public DomainLimits domain
    cdef public GravityPool nodes
    cdef public Splitter export_interaction

    # pointers for particle position and mass
    cdef np.float64_t *x[3], *m

    # varaibles for parallel run
    cdef int parallel

    cdef public LoadBalance load_bal
    cdef public Splitter import_interaction
    cdef public CarrayContainer remote_nodes

    cdef public buffer_id
    cdef public buffer_pid
    cdef public buffer_import
    cdef public buffer_export

    cdef public np.ndarray send_cnts
    cdef public np.ndarray send_disp
    cdef public np.ndarray recv_cnts
    cdef public np.ndarray recv_disp

    # allocate node functions
    cdef inline void create_root(self)
    cdef inline int get_index(self, int parent_index, np.float64_t x[3])
    cdef inline Node* create_child(self, int parent_index, int child_index)
    cdef inline void create_children(self, int parent_index)

    # tree build functions
    cdef void _build_top_tree(self)
    cdef void _create_top_tree(self, int node_index, LoadNode* load_parent,
            np.int32_t* node_map)
    cdef inline int _leaf_index_toptree(self, np.int64_t key)
    #cdef void _build_tree(self, CarrayContainer pc)

    # moment calculation functions
    cdef void _update_moments(self, int current, int sibling)
    cdef void _export_import_remote_nodes(self)
    cdef void _update_remote_moments(self, int current)

#    # tree walk functions
#    #cdef void _walk(self, Interaction interaction, CarrayContainer pc)
