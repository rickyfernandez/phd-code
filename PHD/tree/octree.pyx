import numpy as np

cimport numpy as np
cimport libc.stdlib as stdlib
cimport cython

from libcpp.set cimport set
from PHD.hilbert.hilbert import hilbert_key_2d


cdef struct LeafToSend:

    np.int64_t particle_index_start
    np.uint64_t number_particles
    int process


cdef struct Node:

    np.uint64_t sfc_key      # space filling curve key for node
    np.uint32_t level        # level of tree

    np.uint64_t sfc_start_key         # first key in space filling curve cut
    np.uint64_t number_sfc_keys       # total number of possible space filling keys in this node

    np.int64_t  particle_index_start  # index of first particle in space filling curve cut
    np.uint64_t number_particles      # number of particles in cut

    int leaf                 # is this node a leaf

    np.float64_t box_length  # side length of node
    np.float64_t center[2]   # center of coordinates of node

    Node* parent             # parent of node
    Node* children           # children nodes, 4 of them
    int   children_index[4]   # index to point to the right child


cdef class QuadTree:


    cdef readonly np.ndarray sorted_particle_keys     # hilbert keys of the particles in key order
    cdef readonly np.uint64_t max_leaf_particles      # criteria to subdivide a node 
    cdef readonly int order                           # number of bits per dimension
    cdef readonly int process                         # rank of process
    cdef readonly int number_process                  # number of process 
    cdef readonly int number_nodes                    # number of created nodes

    cdef Node* root                                   # pointer to the root of the tree
    cdef np.uint32_t xmin, xmax, ymin, ymax           # boundaries of the tree
    cdef np.uint64_t local_number_particles           # local number of particles in process


    def __init__(self, sorted_particle_keys, max_leaf_particles, order, process, number_process):

        self.sorted_particle_keys = np.ascontiguousarray(sorted_particle_keys, dtype=np.int64)
        self.order = order

        # size of the domain is the size of hilbert space
        self.xmin = self.ymin = 0
        self.xmax = self.ymax = 2**(order)

        self.process = process
        self.number_process = number_process

        self.local_number_particles = sorted_particle_keys.shape[0]
        self.max_leaf_particles = max_leaf_particles

        # build root of tree
        self.root = <Node*>stdlib.malloc(sizeof(Node))
        if self.root == <Node*> NULL:
            raise MemoryError

        self.number_nodes = 1

        # create root node which holds all hilbert possible keys in a
        # grid of 2^order resolution per dimension
        self.root.children = NULL
        self.root.sfc_start_key = 0
        self.root.number_sfc_keys = 2**(2*order)
        self.root.particle_index_start = 0
        self.root.number_particles = sorted_particle_keys.shape[0]
        self.root.level = 0
        self.root.leaf = 1
        self.root.box_length = 2**(order)

        # the center of the root is the center of the grid of hilbert keys
        cdef int i
        for i in range(2):
            self.root.center[i] = 0.5*2**(order)


    def build_tree(self):
        self._fill_particles_in_node(self.root, self.max_leaf_particles)


    cdef void _create_node_children(self, Node* node):

        # create children nodes
        node.children = <Node*>stdlib.malloc(sizeof(Node)*4)
        if node.children == <Node*> NULL:
            raise MemoryError

        self.number_nodes += 4

        # pass parent data to children 
        cdef int i, j
        for i in range(4):

            if node.number_sfc_keys < 4:
                raise RuntimeError("Not enough hilbert keys to be split")

            node.children[i].parent = node
            node.children[i].leaf = 1
            node.children[i].level = node.level + 1

            # each child has a cut of hilbert keys from parent
            node.children[i].number_sfc_keys = node.number_sfc_keys/4
            node.children[i].sfc_start_key   = node.sfc_start_key + i*node.number_sfc_keys/4

            node.children[i].particle_index_start = -1
            node.children[i].number_particles = 0

            node.children[i].box_length = node.box_length/2.0
            node.children[i].children = NULL


        # create children center coordinates by shifting parent coordinates by 
        # half box length in each dimension
        cdef np.uint64_t key
        cdef int child_node_index
        for i in range(2):
            for j in range(2):

                # compute hilbert key for each child
                key = hilbert_key_2d( <np.uint32_t> (node.center[0] + (2*i-1)*node.box_length/4.0),
                        <np.uint32_t> (node.center[1] + (2*j-1)*node.box_length/4.0), self.order)

                # find which node this key belongs to it and store the key
                # center coordinates
                child_node_index = (key - node.sfc_start_key)/(node.number_sfc_keys/4)
                node.children[child_node_index].sfc_key = key
                node.children[child_node_index].center[0] = node.center[0] + (2*i-1)*node.box_length/4.0
                node.children[child_node_index].center[1] = node.center[1] + (2*j-1)*node.box_length/4.0

                # the children are in hilbert order, this mapping allows to grab children
                # in bottom-left, upper-left, bottom-right, upper-right order
                node.children_index[(i<<1) + j] = child_node_index

        # parent is no longer a leaf
        node.leaf = 0


    cdef void _fill_particles_in_node(self, Node* node, np.uint64_t max_leaf_particles):

        cdef int i, child_node_index

        self._create_node_children(node)

        # loop over parent particles and assign them to proper child
        for i in range(node.particle_index_start, node.particle_index_start + node.number_particles):

            # which node does this particle belong to
            child_node_index = (self.sorted_particle_keys[i] - node.sfc_start_key)/(node.number_sfc_keys/4)

            # if child node is empty then this is the first particle in the cut
            if node.children[child_node_index].number_particles == 0:
                node.children[child_node_index].particle_index_start = i

            # update the number of particles for child
            node.children[child_node_index].number_particles += 1


        # if child has more particles then the maximum allowed, then subdivide 
        for i in range(4):
            if node.children[i].number_particles > max_leaf_particles:
                self._fill_particles_in_node(&node.children[i], max_leaf_particles)


    cdef void _count_leaves_with_particles(self, Node* node, int* num):

        cdef int i
        if node.children == NULL:
            if node.number_particles != 0:
                num[0] += 1
        else:
            for i in range(4):
                self._count_leaves_with_particles(&node.children[i], num)


    def count_leaves_with_particles(self):
        cdef int num = 0
        self._count_leaves_with_particles(self.root, &num)
        return num


    cdef void _collect_keys_levels_for_export(self, Node* node, np.uint64_t *keys, np.uint32_t *levels,
            np.float64_t *x, np.float64_t *y, np.float64_t *box_lengths, int* counter):

        cdef int i
        if node.children == NULL:
            if node.number_particles != 0:

                keys[counter[0]]   = node.sfc_key
                levels[counter[0]] = node.level
                box_lengths[counter[0]] = node.box_length

                x[counter[0]] = node.center[0]
                y[counter[0]] = node.center[1]

                counter[0] += 1

        else:
            for i in range(4):
                self._collect_keys_levels_for_export(&node.children[i], keys, levels, x, y,\
                        box_lengths, counter)


    def collect_leaves_with_particles_for_export(self):

        cdef int num = 0
        cdef int counter = 0

        # count the number of leaves with particles in the tree 
        self._count_leaves_with_particles(self.root, &num)
        #return num_leaves

        cdef np.uint64_t[:]  keys   = np.empty(num, dtype=np.uint64)
        cdef np.uint32_t[:]  levels = np.empty(num, dtype=np.uint32)

        cdef np.float64_t[:] x = np.empty(num, dtype=np.float64)
        cdef np.float64_t[:] y = np.empty(num, dtype=np.float64)

        cdef np.float64_t[:] box_width = np.empty(num, dtype=np.float64)

        self._collect_keys_levels_for_export(self.root, &keys[0], &levels[0], &x[0], &y[0],\
                &box_width[0], &counter)

        # for debugging i'm returning x, y coords and box width
        return np.asarray(keys), np.asarray(levels), np.asarray(x), np.asarray(y), np.asarray(box_width)


#    def coolect_boundary_particles(self):


#    cdef void _build_partial_tree(self):
#        cdef np.uint64_t max_leaf = <np.uint64_t> (0.1*self.local_number_particles/self.number_process)
#        self._fill_particles_in_node(self.root, max_leaf)



#    def fill_process_leaves(self, np.uint64_t[:] keys, np.uint32[:] level, int p):
#
#        Node *node
#        for i in xrange(keys.shap[0]):
#            node = self._find_node_by_key_level(key, level)
#            if node.level == level:
#                node = create_process_node(key, level)
#            node.process = p




#    cdef *Node create_process_nodes(self, Node* node, key, level):
#
#        _create_node_children(o)
#
#        # which node does this key belong to
#        child_node_index = (keys - node.sfc_start_key)/(node.number_sfc_keys/4)
#
#        if o.children[child_node_index].level != level:
#            return self._fill_particles_in_node(&o.children[child_node_index], level)
#        else:
#            return &o.children[child_node_index]




    cdef Node* _find_node_by_key(self, np.uint64_t key):

        cdef Node* candidate = self.root
        cdef int child_node_index

        while candidate.leaf != 1:
            child_node_index = (key - candidate.sfc_start_key)/(candidate.number_sfc_keys/4)
            candidate = &candidate.children[child_node_index]

        return candidate


    cdef Node* _find_node_by_key_level(self, np.uint64_t key, np.uint32_t level):

        cdef Node* candidate = self.root
        cdef int child_node_index

        while candidate.level < level and candidate.leaf != 1:
            child_node_index = (key - candidate.sfc_start_key)/(candidate.number_sfc_keys/4)
            candidate = &candidate.children[child_node_index]

        return candidate


    cdef void _subneighbor_find(self, list node_list, Node* candidate, int i, int j):

        if i == j == 1: return

        cdef Node* child_cand
        cdef np.int64_t num_loop[2], index[2], off[2][2], ii, ij

        index[0] = i
        index[1] = j

        # num_steps and walk?
        for ii in range(2):

            # no offset 
            if index[ii] == 1:
                num_loop[ii] = 2
                off[ii][0] = 0
                off[ii][1] = 1

            # left offset
            elif index[ii] == 0:
                num_loop[ii] = 1
                off[ii][0] = 1

            # right offset
            elif index[ii] == 2:
                num_loop[ii] = 1
                off[ii][0] = 0

        for ii in range(num_loop[0]):
            for ij in range(num_loop[1]):

                child_index = (off[0][ii] << 1) + off[1][ij]
                child_cand = &candidate.children[candidate.children_index[child_index]]

                if child_cand.children != NULL:
                    self._subneighbor_find(node_list, child_cand, i, j)
                else:
                    node_list.append([child_cand.center[0], child_cand.center[1], child_cand.box_length,\
                            child_cand.level, child_cand.particle_index_start, child_cand.number_particles])


    def node_neighbor_search(self, np.uint64_t key):

        cdef Node *node, *neighbor
        cdef np.uint64_t node_key, neighbor_node_key
        cdef np.int64_t x, y
        cdef list node_list = []
        cdef int i, j

        #cdef set node_set = set()
        cdef set[np.uint64_t] node_key_set
        cdef set[np.uint64_t].iterator it

        # find the node leaf that the search particle lives in
        node = self._find_node_by_key(key)

        # find neighbor node by shifting leaf node key by half box length
        for i in range(3):
            for j in range(3):

                x = <np.int64_t> (node.center[0] + (i-1)*node.box_length)
                y = <np.int64_t> (node.center[1] + (j-1)*node.box_length)

                if i == j == 1:
                    continue

                if (self.xmin <= x and x <= self.xmax) and (self.ymin <= y and y <= self.ymax):

                    neighbor_node_key = hilbert_key_2d(x, y, self.order)

                    # find neighbor node that is at max the same level of query node
                    neighbor = self._find_node_by_key_level(neighbor_node_key, node.level)

                    # make sure we don't add duplicate neighbors
                    if node_key_set.find(neighbor.sfc_key) == node_key_set.end():
                    #if neighbor.sfc_key not in node_set:

                        #node_set.add(neighbor.sfc_key)
                        node_key_set.insert(neighbor.sfc_key)

                        # check if their are sub nodes, if so collet them too
                        if neighbor.children != NULL:
                            self._subneighbor_find(node_list, neighbor, i, j)
                        else:
                            node_list.append([neighbor.center[0], neighbor.center[1], neighbor.box_length,\
                                    neighbor.level, neighbor.particle_index_start, neighbor.number_particles])

        return node_list


    # temporary function to do outputs in python
    def find_node(self, key):

        cdef Node* node = self._find_node_by_key(key)
        return [node.center[0], node.center[1], node.box_length,
            node.level, node.particle_index_start, node.number_particles]


    # temporary function to do outputs in python
    cdef _iterate(self, Node* node, list data_list):

        data_list.append([node.center[0], node.center[1], node.box_length,
            node.level, node.particle_index_start, node.number_particles])

        cdef int i
        if node.children != NULL:
            for i in range(4):
                self._iterate(&node.children[i], data_list)


    # temporary function to do outputs in python
    def dump_data(self):
        cdef list data_list = []
        self._iterate(self.root, data_list)
        return data_list


    # temporary function to do outputs in python
    cdef void _free_nodes(self, Node* node):
        cdef int i
        if node.children != NULL:
            for i in range(4):
                self._free_nodes(&node.children[i])
            stdlib.free(node.children)


    def __dealloc__(self):
        self._free_nodes(self.root)
        stdlib.free(self.root)

# --- this needs to be worked on later --- it improves neighbor search
#                    # starting key and oct
#                    oct_key = p.sfc_key
#                    ancestor_oct = p
#
#                    while oct_key != neighbor_oct_key:
#
#                        # shift keys until they are equal, this happens
#                        # at the common ancestor oct
#                        oct_key = (oct_key << 2)
#                        neighbor_oct_key = (neighbor_oct_key << 2)
#
#                        # go up the parent oct
#                        ancestor_oct = ancestor_oct.parent
