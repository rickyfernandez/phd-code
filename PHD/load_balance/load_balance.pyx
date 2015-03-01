import numpy as np

cimport numpy as np
cimport libc.stdlib as stdlib
cimport cython

from PHD.hilbert.hilbert import hilbert_key_2d


cdef struct Node:

    np.uint64_t sfc_start_key            # first key in space filling curve cut in node
    np.uint64_t number_sfc_keys          # total number of possible space filling keys in this node

    np.uint64_t particle_index_start     # index of first particle in space filling curve cut
    np.uint64_t number_particles         # number of particles in cut
    np.uint64_t number_segments          # number of hilbert cuts

    int leaf                             # is this node a leaf

    Node* children                       # children nodes, 4 of them


cdef class QuadTree:

    cdef readonly np.uint32_t number_nodes  # number of created nodes

    cdef np.uint64_t[:] sorted_keys      # hilbert keys of the particles/segments in order
    cdef np.uint64_t[:] num_part_leaf    # if using segments, then this number of particles in segment 

    cdef int order                       # number of bits per dimension
    cdef double factor                   #  
    cdef int use_num_part_leaf           # flag tree built from hilbert cuts 

    cdef int total_num_process           # global total number of process
    cdef np.uint64_t total_num_part      # global total number of particles

    cdef np.uint64_t max_in_leaf         # max allowed particles in a leaf

    cdef Node* root                      # pointer to the root of the tree

    def __init__(self, np.uint64_t total_num_part, np.ndarray[np.uint64_t, ndim=1] sorted_keys,
            np.ndarray[np.uint64_t, ndim=1] num_part_leaf=None, int total_num_process=1,
            double factor=0.1, int order=21):

        self.order = order
        self.factor = factor
        self.sorted_keys = sorted_keys
        self.total_num_part = total_num_part
        self.total_num_process = total_num_process

        # are we building tree from hilbert segments
        if num_part_leaf != None:
            self.num_part_leaf = num_part_leaf
            self.use_num_part_leaf = 1
            self.max_in_leaf = <np.uint64_t> (factor*total_num_part/total_num_process)

        # building tree from particles
        else:
            self.use_num_part_leaf = 0
            self.max_in_leaf = <np.uint64_t> (factor*total_num_part/total_num_process**2)



    def build_tree(self, max_in_leaf=None):

        cdef np.uint64_t max_leaf
        if max_in_leaf != None:
            max_leaf = max_in_leaf
        else:
            max_leaf = self.max_in_leaf

        self.root = <Node*> stdlib.malloc(sizeof(Node))
        if self.root == <Node*> NULL:
            raise MemoryError

        self.number_nodes = 1

        # create root node which holds all possible hilbert
        # keys in a grid of 2^order resolution per dimension
        self.root.children = NULL
        self.root.sfc_start_key = 0
        self.root.number_sfc_keys = 2**(2*self.order)
        self.root.particle_index_start = 0
        self.root.leaf = 1

        if self.use_num_part_leaf:
            self.root.number_particles = self.total_num_part
            self.root.number_segments  = self.sorted_keys.shape[0]
            self._fill_segments_nodes(self.root, max_leaf)
        else:
            self.root.number_particles = self.sorted_keys.shape[0]
            self.root.number_segments = 0
            self._fill_particles_nodes(self.root, max_leaf)


    cdef void _create_node_children(self, Node* node):

        # create children nodes
        node.children = <Node*>stdlib.malloc(sizeof(Node)*4)
        if node.children == <Node*> NULL:
            raise MemoryError

        self.number_nodes += 4

        # pass parent data to children 
        cdef int i #, j
        for i in range(4):

            if node.number_sfc_keys < 4:
                raise RuntimeError("Not enough hilbert keys to be split")

            node.children[i].leaf = 1

            # each child has a cut of hilbert keys from parent
            node.children[i].number_sfc_keys = node.number_sfc_keys/4
            node.children[i].sfc_start_key   = node.sfc_start_key + i*node.number_sfc_keys/4
            node.children[i].particle_index_start = node.particle_index_start
            node.children[i].number_particles = 0
            node.children[i].number_segments = 0
            node.children[i].children = NULL

        # parent is no longer a leaf
        node.leaf = 0


    cdef void _fill_particles_nodes(self, Node* node, np.uint64_t max_in_leaf):

        cdef int i, child_node_index

        self._create_node_children(node)

        # loop over parent particles and assign them to proper child
        for i in xrange(node.particle_index_start, node.particle_index_start + node.number_particles):

            # which node does this particle/segment belong to
            child_node_index = (self.sorted_keys[i] - node.sfc_start_key)/(node.number_sfc_keys/4)

            # if child node is empty then this is the first particle in the cut
            if node.children[child_node_index].number_particles == 0:
                node.children[child_node_index].particle_index_start = i

            # update the number of particles for child
            node.children[child_node_index].number_particles += 1

        # if child has more particles then the maximum allowed, then subdivide 
        for i in xrange(4):
            if node.children[i].number_particles > max_in_leaf:
                self._fill_particles_nodes(&node.children[i], max_in_leaf)


    cdef void _fill_segments_nodes(self, Node* node, np.uint64_t max_in_leaf):

        cdef int i, child_node_index

        self._create_node_children(node)

        # loop over parent segments and assign them to proper child
        for i in xrange(node.particle_index_start, node.particle_index_start + node.number_segments):

            # which node does this segment belong to
            child_node_index = (self.sorted_keys[i] - node.sfc_start_key)/(node.number_sfc_keys/4)

            if node.children[child_node_index].number_segments == 0:
                node.children[child_node_index].particle_index_start = i

            # update the number of particles for child
            node.children[child_node_index].number_particles += self.num_part_leaf[i]
            node.children[child_node_index].number_segments += 1

        # if child has more particles then the maximum allowed, then subdivide 
        for i in xrange(4):
            if node.children[i].number_particles > max_in_leaf:
                self._fill_segments_nodes(&node.children[i], max_in_leaf)


    cdef void _count_leaves(self, Node* node, np.uint32_t* num):

        cdef int i
        if node.children == NULL:
            num[0] += 1
        else:
            for i in xrange(4):
                self._count_leaves(&node.children[i], num)


    def count_leaves(self):
        cdef np.uint32_t num = 0
        self._count_leaves(self.root, &num)
        return num


    cdef void _collect_leaves_for_export(self, Node* node, np.uint64_t *start_keys, np.uint32_t *num_part_leaf,
            np.uint32_t* counter):

        cdef int i
        if node.children == NULL:

            start_keys[counter[0]]    = node.sfc_start_key
            num_part_leaf[counter[0]] = node.number_particles
            counter[0] += 1

        else:
            for i in range(4):
                self._collect_leaves_for_export(&node.children[i], start_keys, num_part_leaf, counter)


    def collect_leaves_for_export(self):

        cdef np.uint32_t num = 0
        cdef np.uint32_t counter = 0

        self._count_leaves(self.root, &num)

        cdef np.uint64_t[:]  start_keys    = np.empty(num, dtype=np.uint64)
        cdef np.uint32_t[:]  num_part_leaf = np.empty(num, dtype=np.uint32)

        self._collect_leaves_for_export(self.root, &start_keys[0], &num_part_leaf[0], &counter)

        return np.asarray(start_keys), np.asarray(num_part_leaf)


    cdef void _free_nodes(self, Node* node):
        cdef int i
        if node.children != NULL:
            for i in range(4):
                self._free_nodes(&node.children[i])
            stdlib.free(node.children)


    def __dealloc__(self):
        self._free_nodes(self.root)
        stdlib.free(self.root)


cdef class LoadBalance:

#    cdef object comm
    cdef QuadTree global_tree
    cdef int order
    cdef np.uint64_t number_particles

    cdef readonly np.ndarray keys
    cdef readonly np.ndarray sorted_keys


#    def __init__(self, particles, comm, order=21):
    def __init__(self, np.ndarray[np.float64_t, ndim=2] particles, int order=21):

        self.order = order
        self.number_particles = particles.shape[1]

        self.keys = np.empty(self.number_particles, dtype=np.uint64)
        self.sorted_keys = np.empty(self.number_particles, dtype=np.uint64)

        cdef i
        for i in xrange(self.number_particles):
            self.sorted_keys[i] = self.keys[i] = hilbert_key_2d(
                    <np.uint32_t> (particles[0,i]*2**self.order),
                    <np.uint32_t> (particles[1,i]*2**self.order),
                    self.order)

        self.sorted_keys.sort()


#    def decomposition(self):
#
#        # generate hilbert keys for each particle
#
#
#    cdef _get_bounding_box(self):
#        pass
#
#    cdef void _build_global_tree(self):
#
#        # collect number of particles from all process
#        # construct local tree
#        cdef QuadTree local_tree = QuadTree(self.sorted_keys, order=self.order)
#        cdef int max_part_in_leaves = factor*N/p**2
#
#        local_tree.build_tree(max_part_in_leaves)
#
#        # collect leaves to send to all process
#        leaf_keys, num_part_leaf = local_tree.collect_leaves_for_export()
#
#        # sort the hilbert segments
#        ind = self.leaf_keys.argsort()
#        self.leaf_keys = self.leaf_keys[ind]
#        self.num_part_leaf   = self.num_part_leaf[ind]
#
#
#        # export all leaves
#
#        # rebuild tree using global leaves
#        self.global_tree = QuadTree(leaf_keys, num_part_leaf, total_num_particles, self.order)
#        max_part_in_leaves = factor*N/p
#
#        self.global_tree.build(max_part_in_leaves)
#
#
#    cdef _find_work_done(self):
#        pass
#
#    cdef _find_split_in_work(self):
#        pass
