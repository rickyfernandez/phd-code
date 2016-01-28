
cimport numpy as np
#from libc.posix cimport off_t
from containers.containers cimport ParticleContainer

ctypedef np.int64_t (*hilbert_type)(np.int32_t, np.int32_t, np.int32_t, int)

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
    #off_t children_start          # first child offset form parent for pointer arithmetic
    int children_start          # first child offset form parent for pointer arithmetic

    int   children_index[8]     # index to point to the right child

cdef class TreeMemoryPool:

    cdef int used                       # number of nodes used in the pool
    cdef int capacity                   # total capacity of the pool

    cdef Node* node_array               # array holding all nodes for use

    cdef Node* get(self, int count)     # return 'count' many nodes
    cdef void resize(self, int size)    # resize array of nodes to length size
    cdef void reset(self)               # reset the pool

cdef class BaseTree:

    cdef np.int64_t[:] sorted_part_keys    # hilbert keys of the particles/segments in order
    cdef np.int64_t[:] sorted_segm_keys    # hilbert keys of the particles/segments in order
    cdef np.int32_t[:] num_part_leaf       # if using segments, then this number of particles in segment 

    cdef np.float64_t[:] domain_corner     # corner of particle domain
    cdef double domain_length              # particle domain size
    cdef double domain_fac                 # factor for domain to hilbert space mapping

    cdef int order                         # number of bits per dimension
    cdef double factor                     #  
    cdef int build_using_cuts              # flag tree built from hilbert cuts 
    cdef int total_num_process             # global total number of process
    cdef int total_num_part                # global total number of particles
    cdef int max_in_leaf                   # max allowed particles in a leaf
    cdef int number_leaves                 # number of leaves

    cdef int dim                           # dimension of the problem
    cdef hilbert_type hilbert_func         # hilbert key generator

    cdef TreeMemoryPool mem_pool           # pool of nodes
    cdef Node* root                        # pointer to the root of the tree

    cdef void _assign_leaves_to_array(self, Node* node)
    cdef void _create_node_children(self, Node* node)
    cdef void _fill_particles_nodes(self, Node* node, int max_in_leaf)
    cdef void _fill_segments_nodes(self, Node* node, np.uint64_t max_in_leaf)
    cdef void _count_leaves(self, Node* node)
    cdef void _collect_leaves_for_export(self, Node* node, np.int64_t *start_keys,
            np.int32_t *num_part_leaf, int* counter)

    cdef Node* _find_leaf(self, np.int64_t key)
    cdef Node* _find_node_by_key_level(self, np.uint64_t key, np.uint32_t level)
    cdef void _node_neighbor_search(self, Node* node, int index[3], ParticleContainer part_array, np.int32_t* leaf_proc,
            set boundary_keys, set leaf_neighbors, int rank)
    cdef void _subneighbor_find(self, Node* candidate, int index[3], np.int32_t* leaf_proc, ParticleContainer part_array,
            set boundary_keys, int rank)

cdef class QuadTree(BaseTree):

    cdef np.float64_t xmin, xmax, ymin, ymax

    cdef void _subneighbor_find(self, Node* candidate, int index[3], np.int32_t* leaf_proc, ParticleContainer part_array,
            set boundary_keys, int rank)

cdef class OcTree(BaseTree):

    cdef np.float64_t xmin, xmax, ymin, ymax, zmin, zmax

    cdef void _subneighbor_find(self, Node* candidate, int index[3], np.int32_t* leaf_proc, ParticleContainer part_array,
            set boundary_keys, int rank)
