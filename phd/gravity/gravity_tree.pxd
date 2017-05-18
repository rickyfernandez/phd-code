cimport numpy as np
from libcpp.map cimport map

from ..utils.carray cimport LongArray
from .interaction cimport Interaction
from ..domain.domain cimport DomainLimits
from .gravity_pool cimport Node, GravityPool
from ..load_balance.tree cimport Node as LoadNode
from ..containers.containers cimport CarrayContainer
from ..load_balance.load_balance cimport LoadBalance


# tree flags
cdef enum:
    NOT_EXIST = -1
    ROOT = 0
    ROOT_SIBLING = -1
    LEAF = 0x01
    HAS_PARTICLE = 0x02
    TOP_TREE = 0x04
    TOP_TREE_LEAF = 0x08
    TOP_TREE_LEAF_REMOTE = 0x10
    SKIP_BRANCH = 0x20

cdef class GravityTree:

    cdef public CarrayContainer pc              # referecne to particles
    cdef public int number_nodes                # max number of children nodes
    cdef public int dim, rank, size
    cdef public DomainLimits domain             # simulation domain

    cdef public str split_type                  # method to open nodes
    cdef public GravityPool nodes               # node array for gravity tree
    cdef public int calc_potential              # flag if potential is calculated
    cdef public double barnes_angle             # angle to open node in barnes hut
    cdef public Interaction export_interaction  # acceleration calculator

    # pointers for particle position and mass
    cdef np.float64_t *x[3], *m

    # varaibles for parallel run
    cdef int parallel                           # signal if run is in parallel
    cdef public int max_buffer_size             # max number of particles in buffer

    cdef map[int, int]  node_index_to_array

    cdef public LoadBalance load_bal            # reference to load balance
    cdef public Interaction import_interaction  # acceleration calculator
    cdef public CarrayContainer remote_nodes    # container of remote nodes

    cdef public np.ndarray flag_pid             # flag if particle is export to processor
    cdef public LongArray buffer_id             # particle id for export particle
    cdef public LongArray buffer_pid            # processor to send export particle 
    cdef public CarrayContainer buffer_export   # container of particles to import
    cdef public CarrayContainer buffer_import   # container of particles to export

    cdef public np.ndarray send_cnts            # send counts for mpi
    cdef public np.ndarray send_disp            # send displacments for mpi
    cdef public np.ndarray recv_cnts            # receive counts for mpi
    cdef public np.ndarray recv_disp            # receive counts for mpi

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

    # tree walk functions
    cdef void _serial_walk(self, Interaction interaction)
    cdef void _parallel_walk(self, Interaction interaction)
    cdef void _import_walk(self, Interaction interaction)
    cdef void _export_walk(self, Interaction interaction)
