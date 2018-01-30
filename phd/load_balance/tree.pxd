cimport numpy as np

from ..utils.carray cimport LongArray
from ..containers.containers cimport CarrayContainer

ctypedef np.int64_t (*hilbert_type)(np.int32_t, np.int32_t, np.int32_t, int)

cdef extern from "stdlib.h":
    void qsort(void *array, size_t count, size_t size,
            int (*compare)(const void *, const void *))

cdef int int_cmp(const void *pa, const void *pb)

# forward decleration
cdef struct Node

cdef struct Node:

    np.int64_t sfc_key          # space filling curve key for node
    np.int64_t sfc_start_key    # first key in space filling curve cut in node
    np.int64_t number_sfc_keys  # total number of possible space filling keys in this node

    np.int64_t level            # level of tree
    np.float64_t box_length     # side length of node
    np.float64_t center[3]      # center coordinates of node

    int particle_index_start    # index of first particle in space filling curve cut
    int number_particles        # number of particles in cut
    int number_segments         # number of hilbert cuts
    int leaf                    # is this node a leaf
    int array_index             # index of global array that stores leaf data
    int children_start          # first child offset form parent for pointer arithmetic

    #int children_index[8]       # index to point to the right child
    int zorder_to_hilbert[8]    # map zorder to hilbert 

cdef class TreeMemoryPool:

    cdef int used                       # number of nodes used in the pool
    cdef int capacity                   # total capacity of the pool

    cdef Node* node_array               # array holding all nodes

    cdef Node* get(self, int count)     # return 'count' many nodes
    cdef void resize(self, int size)    # resize array of nodes to length size
    cdef void reset(self)               # reset the pool
    cpdef int number_leaves(self)       # number of leves in tree
    cpdef int number_nodes(self)        # number of nodes in tree

cdef class Tree:

    cdef double domain_corner[3]           # corner of particle domain
    cdef double domain_length              # particle domain size
    cdef double domain_fac                 # factor for domain to hilbert space mapping

    cdef int order                         # number of bits per dimension
    cdef int min_in_leaf
    cdef double factor                     # fraction of particles in max leaf
    cdef int total_num_part                # global total number of particles

    cdef int number_leaves

    cdef int dim                           # dimension of the problem
    cdef hilbert_type hilbert_func         # hilbert key generator

    cdef int bounds[2][3]                  # min and max of boundary

    cdef public TreeMemoryPool mem_pool    # pool of nodes
    cdef Node* root                        # pointer to the root of the tree

    cdef void _leaves_to_array(self, Node* node, int* num_leaves)
    cdef void _create_node_children(self, Node* node)
    cdef void _fill_particles_nodes(self, Node* node, np.int64_t* sorted_part_keys, int max_in_leaf)
    cdef void _fill_segments_nodes(self, Node* node, np.int64_t* sorted_segm_keys,
            np.int32_t* sorted_segm_parts, int max_in_leaf)

    cpdef _build_local_tree(self, np.ndarray[np.int64_t, ndim=1] sorted_part_keys, int max_in_leaf)
    cdef void _build_global_tree(self, int global_num_particles, np.ndarray[np.int64_t, ndim=1] sorted_segm_keys,
            np.ndarray[np.int32_t, ndim=1]  sorted_segm_parts, int max_in_leaf)

    cdef void construct_global_tree(self, CarrayContainer pc, object comm)

    cdef Node* find_leaf(self, np.int64_t key)
    cdef int get_nearest_process_neighbors(self, double center[3], double h,
            LongArray leaf_pid, int rank, LongArray nbrs)
    cdef void _neighbors(self, Node* node, double smin[3], double smax[3], np.int32_t* leaf_proc, int rank, LongArray nbrs)
    cdef int get_nearest_intersect_process_neighbors(self, double center[3], double old_h,
            double new_h, LongArray leaf_pid, int rank, LongArray nbrs)
