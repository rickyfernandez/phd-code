
cimport numpy as np
from particles.particle_array cimport ParticleArray

# forward decleration
cdef struct Node

cdef struct Node:

    np.int64_t sfc_key          # space filling curve key for node
    np.int64_t sfc_start_key    # first key in space filling curve cut in node
    np.int64_t number_sfc_keys  # total number of possible space filling keys in this node

    np.int64_t level            # level of tree
    np.float64_t box_length     # side length of node
    np.float64_t center[2]      # center coordinates of node

    int particle_index_start    # index of first particle in space filling curve cut
    int number_particles        # number of particles in cut
    int number_segments         # number of hilbert cuts
    int leaf                    # is this node a leaf
    int array_index             # index of global array that stores leaf data

    Node* children              # children nodes, 4 of them
    int   children_index[4]     # index to point to the right child

cdef class QuadTree:

    cdef np.int64_t[:] sorted_part_keys    # hilbert keys of the particles/segments in order
    cdef np.int64_t[:] sorted_segm_keys    # hilbert keys of the particles/segments in order
    cdef np.int32_t[:] num_part_leaf       # if using segments, then this number of particles in segment 

    cdef int order                         # number of bits per dimension
    cdef double factor                     #  
    cdef int build_using_cuts              # flag tree built from hilbert cuts 
    cdef int total_num_process             # global total number of process
    cdef int total_num_part                # global total number of particles
    cdef int max_in_leaf                   # max allowed particles in a leaf
    cdef int number_leaves                 # number of leaves
    cdef int number_nodes                  # number of created nodes

    cdef Node* root                        # pointer to the root of the tree
    cdef np.float64_t xmin, xmax, ymin, ymax

    cdef void _assign_leaves_to_array(self, Node* node)
    cdef void _create_node_children(self, Node* node)
    cdef void _fill_particles_nodes(self, Node* node, int max_in_leaf)
    cdef void _fill_segments_nodes(self, Node* node, np.uint64_t max_in_leaf)
    cdef void _count_leaves(self, Node* node)
    cdef void _collect_leaves_for_export(self, Node* node, np.int64_t *start_keys,
            np.int32_t *num_part_leaf, int* counter)

    cdef Node* _find_leaf(self, np.int64_t key)
    cdef Node* _find_node_by_key_level(self, np.uint64_t key, np.uint32_t level)
    cdef void _create_boundary_particles(self, Node* node, ParticleArray part_array, np.int32_t* leaf_proc,
            set boundary_keys, int rank)
    cdef void node_neighbor_search(self, Node* node, ParticleArray part_array, np.int32_t* leaf_proc,
            set boundary_keys, int rank)
    cdef void _subneighbor_find(self, Node* candidate, np.int32_t* leaf_proc, ParticleArray part_array,
            set boundary_keys, int rank, int i, int j)
    cdef void _iterate(self, Node* node, list data_list)
    cdef void _free_nodes(self, Node* node)
