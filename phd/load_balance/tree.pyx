import numpy as np

cimport numpy as np
cimport libc.stdlib as stdlib
cimport cython

from hilbert.hilbert import hilbert_key_2d


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

    def __init__(self, int total_num_part,
            np.ndarray[np.int64_t, ndim=1] sorted_part_keys,
            np.ndarray[np.int64_t, ndim=1] sorted_segm_keys=None,
            np.ndarray[np.int32_t, ndim=1] num_part_leaf=None,
            int total_num_process=1, double factor=1.0, int order=21):

        self.order = order
        self.factor = factor
        self.sorted_part_keys = sorted_part_keys
        self.total_num_part = total_num_part
        self.total_num_process = total_num_process
        self.root = NULL

        # size of the domain is the size of hilbert space
        self.xmin = self.ymin = 0
        self.xmax = self.ymax = 2**(order)

        # building tree from particles
        if sorted_segm_keys is None:
            self.build_using_cuts = 0
            self.max_in_leaf = <int> (factor*total_num_part/total_num_process**2)

        # building tree from hilbert segments
        else:
            self.build_using_cuts = 1
            self.num_part_leaf = num_part_leaf
            self.sorted_segm_keys = sorted_segm_keys
            self.max_in_leaf = <int> (factor*total_num_part/total_num_process)

    def assign_leaves_to_array(self):
        """
        Map each leaf to an array index.

        Returns
        -------
        number_leaves : int
            Number of leaves in the tree.
        """
        self.number_leaves = 0
        self._assign_leaves_to_array(self.root)

        return self.number_leaves

    cdef void _assign_leaves_to_array(self, Node* node):
        """
        Recursively walk the tree mapping each leaf to an array index.

        Paramters
        --------
        node : Node
        """
        cdef int i
        if node.children == NULL:
            node.array_index = self.number_leaves
            self.number_leaves += 1
        else:
            for i in xrange(4):
                self._assign_leaves_to_array(&node.children[i])

    def build_tree(self, max_in_leaf=None):
        """
        Create a tree by recursively subdividing hilbert cuts.

        Parameters
        ----------
        max_in_leaf : int
            max number of particles in a node, default is factor * total number
            of particles / number of process
        """
        cdef int max_leaf
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
        self.root.level = 0
        self.root.box_length = 2**(self.order)

        cdef int i
        for i in range(2):
            # the center of the root is the center of the grid of hilbert space
            self.root.center[i] = 0.5*2**(self.order)

        if self.build_using_cuts:
            self.root.number_particles = self.total_num_part
            self.root.number_segments  = self.sorted_segm_keys.shape[0]
            self._fill_segments_nodes(self.root, max_leaf)
        else:
            self.root.number_particles = self.sorted_part_keys.shape[0]
            self.root.number_segments = 0
            self._fill_particles_nodes(self.root, max_leaf)

    def calculate_work(self, np.int64_t[:] keys, np.int32_t[:] work):
        """
        Calculate the work done by each leaf.

        Parameters
        ----------
        keys : ndarray
            Hilbert key for each real particle.
        work : ndarray
            Array of size number of leaves which stores the work done in
            each leaf.
        """
        cdef Node* node
        cdef int i
        for i in xrange(keys.shape[0]):
            node = self._find_leaf(keys[i])
            work[node.array_index] += 1

    def count_particles_export(self, np.int64_t[:] keys, np.int32_t[:] leaf_procs, int my_proc):
        """
        Loop through real particles and count the number that have to be exported.

        Parameters
        ----------
        keys : ndarray
            Hilbert key for each real particle.
        leaf_procs : ndarray
            Rank of process for each leaf.
        my_proc : int
            Rank of current process.
        """
        cdef Node *node
        cdef int i, proc, count=0
        for i in xrange(keys.shape[0]):
            node = self._find_leaf(keys[i])
            proc = leaf_procs[node.array_index]

            if proc != my_proc:
                count += 1

        return count

    def collect_particles_export(self, np.int64_t[:] keys, np.int32_t[:] part_ids, np.int32_t[:] proc_ids,
            np.int32_t[:] leaf_procs, int my_proc):
        """
        Collect export particle indices and the process that it will be sent too.

        Parameters
        ----------
        keys : ndarray
            Hilbert key for each real particle.
        part_ids : ndarray
            Particle indices that need to be exported.
        proc_ids : ndarray
            The process that each exported particle must be sent too.
        leaf_procs : ndarray
            Rank of process for each leaf.
        my_proc : int
            Rank of current process.

        """
        cdef Node *node
        cdef int i, proc, count=0
        for i in xrange(keys.shape[0]):

            node = self._find_leaf(keys[i])
            proc = leaf_procs[node.array_index]

            if proc != my_proc:
                part_ids[count] = i
                proc_ids[count] = proc
                count += 1

    cdef void _create_node_children(self, Node* node):
        """
        Subdivide node into 4 children and transfer appropriate node
        information to each child.

        Parameters
        ----------
        node : Node
            Node that will be subdivided.
        """
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

            # each child has a cut of hilbert keys from parent
            node.children[i].number_sfc_keys = node.number_sfc_keys/4
            node.children[i].sfc_start_key   = node.sfc_start_key + i*node.number_sfc_keys/4
            node.children[i].particle_index_start = node.particle_index_start

            node.children[i].level = node.level + 1
            node.children[i].box_length = node.box_length/2.0

            node.children[i].number_particles = 0
            node.children[i].number_segments = 0
            node.children[i].children = NULL

        # create children center coordinates by shifting parent coordinates by 
        # half box length in each dimension
        cdef np.int64_t key
        cdef int child_node_index
        for i in range(2):
            for j in range(2):

                # compute hilbert key for each child
                key = hilbert_key_2d( <np.int32_t> (node.center[0] + (2*i-1)*node.box_length/4.0),
                        <np.int32_t> (node.center[1] + (2*j-1)*node.box_length/4.0), self.order)

                # find which node this key belongs to it and store the key
                # center coordinates
                child_node_index = (key - node.sfc_start_key)/(node.number_sfc_keys/4)
                node.children[child_node_index].sfc_key = key
                node.children[child_node_index].center[0] = node.center[0] + (2*i-1)*node.box_length/4.0
                node.children[child_node_index].center[1] = node.center[1] + (2*j-1)*node.box_length/4.0

                # the children are in hilbert order, this mapping allows to grab children
                # in bottom-left, upper-left, bottom-right, upper-right order
                node.children_index[(i<<1) + j] = child_node_index

    cdef void _fill_particles_nodes(self, Node* node, int max_in_leaf):
        """
        Subdivide node and bin each particle to the appropriate child. This function is recrusive
        and will continue untill each child contains less than max_in_leaf particles.

        Parameters
        ----------
        node : Node
            Node that will be subdivided.
        max_in_leaf : int
            max number of particles in a node.
        """
        cdef int i, child_node_index

        self._create_node_children(node)

        # loop over parent particles and assign them to proper child
        for i in xrange(node.particle_index_start, node.particle_index_start + node.number_particles):

            # which node does this particle/segment belong to
            child_node_index = (self.sorted_part_keys[i] - node.sfc_start_key)/(node.number_sfc_keys/4)

            if child_node_index < 0 or child_node_index > 4:
                raise RuntimeError("hilbert key out of bounds")

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
        """
        Subdivide node and bin each hilbert segment to the appropriate child. This function is
        recrusive and will continue untill each child contains less than max_in_leaf particles.

        Parameters
        ----------
        node : Node
            Node that will be subdivided.
        max_in_leaf : int
            max number of particles in a node.
        """
        cdef int i, child_node_index

        self._create_node_children(node)

        # loop over parent segments and assign them to proper child
        for i in xrange(node.particle_index_start, node.particle_index_start + node.number_segments):

            # which node does this segment belong to
            child_node_index = (self.sorted_segm_keys[i] - node.sfc_start_key)/(node.number_sfc_keys/4)

            if child_node_index < 0 or child_node_index > 4:
                raise RuntimeError("hilbert key out of bounds")

            if node.children[child_node_index].number_segments == 0:
                node.children[child_node_index].particle_index_start = i

            # update the number of particles for child
            node.children[child_node_index].number_particles += self.num_part_leaf[i]
            node.children[child_node_index].number_segments += 1

        # if child has more particles then the maximum allowed, then subdivide 
        for i in xrange(4):
            if node.children[i].number_particles > max_in_leaf:
                self._fill_segments_nodes(&node.children[i], max_in_leaf)

    cdef void _count_leaves(self, Node* node):
        """
        Recursively count the number of leaves by walking the tree.

        Parameters
        ----------
        node : Node
        """
        cdef int i
        if node.children == NULL:
            self.number_leaves += 1
        else:
            for i in xrange(4):
                self._count_leaves(&node.children[i])

    def count_leaves(self):
        """
        Count the leaves in the tree.

        Returns
        -------
        number_leaves : int
            Number of leaves in the tree
        """
        self.number_leaves = 0
        self._count_leaves(self.root)
        return self.number_leaves

    def count_nodes(self):
        """
        Count the number of nodes that are in the tree.

        Returns
        -------
        number_nodes : int
            Number of nodes in the tree
        """
        return self.number_nodes

    cdef void _collect_leaves_for_export(self, Node* node, np.int64_t *start_keys,
            np.int32_t *num_part_leaf, int* counter):
        """
        Recursively store the first key in the hilbert cut and the number of particles
        for each leaf.

        Parameters
        ----------
        node : Node
        start_keys : pointer to int64 array
            Array that holds the start key for each leaf
        num_part_leaf : pointer to int32 array
            Array that holds the number of particles for each leaf
        counter : pointer to int
            Array index for start_keys and num_part_leaf
        """
        cdef int i
        if node.children == NULL:

            start_keys[counter[0]] = node.sfc_start_key
            num_part_leaf[counter[0]] = node.number_particles
            counter[0] += 1

        else:
            for i in range(4):
                self._collect_leaves_for_export(&node.children[i], start_keys, num_part_leaf, counter)

    # temp: delete later
    def find_particles_process(self, np.int64_t[:] keys, np.int32_t[:] leaf_proc,
            np.int32_t[:] proc_id):

        cdef Node* node
        cdef int i
        for i in xrange(proc_id.size):
            node = self._find_leaf(keys[i])
            proc_id[i] = leaf_proc[node.array_index]

    def collect_leaves_for_export(self):
        """
        For each leaf store the first key in the hilbert cut and the number of particles.

        Returns
        -------
        start_keys : ndarray
            Array that holds the start key for each leaf
        num_part_leaf : ndarray
            Array that holds the number of particles for each leaf
        """
        cdef int counter = 0

        self.number_leaves = 0
        self._count_leaves(self.root)

        cdef np.int64_t[:]  start_keys    = np.empty(self.number_leaves, dtype=np.int64)
        cdef np.int32_t[:]  num_part_leaf = np.empty(self.number_leaves, dtype=np.int32)

        self._collect_leaves_for_export(self.root, &start_keys[0], &num_part_leaf[0], &counter)

        return np.asarray(start_keys), np.asarray(num_part_leaf)

    cdef Node* _find_leaf(self, np.int64_t key):
        """
        Find leaf that contains given hilbert key.

        Parameters
        ----------
        key : int64
            Hilbert key used for search.

        Returns
        -------
        node : Node
            Leaf that contains hilbert key.
        """
        cdef Node* node
        cdef int child_node_index

        node = self.root
        while node.children != NULL:
            child_node_index = (key - node.sfc_start_key)/(node.number_sfc_keys/4)
            node = &node.children[child_node_index]

        return node

    cdef Node* _find_node_by_key_level(self, np.uint64_t key, np.uint32_t level):
        """
        Find node that contains given hilbert key. The node can be at most *level* down
        the tree.

        Parameters
        ----------
        key : int64
            Hilbert key used for search.
        int : int
            The max depth the node can be in the tree

        Returns
        -------
        node : Node
            Node that contains hilbert key.
        """
        cdef Node* candidate = self.root
        cdef int child_node_index

        while candidate.level < level and candidate.children != NULL:
            child_node_index = (key - candidate.sfc_start_key)/(candidate.number_sfc_keys/4)
            candidate = &candidate.children[child_node_index]

        return candidate

    #def create_boundary_particles(self, int rank, np.int32_t[:] leaf_proc):
    def create_boundary_particles(self, ParticleContainer part_container, int rank, np.int32_t[:] leaf_proc):
        """create boundary ghost particles"""
        #cdef list particles = list()
        cdef set boundary_keys = set()

        self._create_boundary_particles(self.root, &part_container, &leaf_proc[0], boundary_keys, rank)
        #self._create_boundary_particles(self.root, &leaf_proc[0], particles, boundary_keys, rank)
        #return particles

    #cdef _create_boundary_particles(self, Node* node, np.int32_t* leaf_proc, list particles, set boundary_keys, int rank):
    cdef _create_boundary_particles(self, Node* node, ParticleContainer* part_container, np.int32_t* leaf_proc,
            set boundary_keys, int rank):
        cdef int i
        if node.children == NULL:
            # leaf belongs to our domain
            if leaf_proc[node.array_index] == rank:
                #self.node_neighbor_search(node, leaf_proc, particles, boundary_keys, rank)
                self.node_neighbor_search(node, part_container, leaf_proc, boundary_keys, rank)
        else:
            for i in range(4):
                self._create_boundary_particles(&node.children[i], part_container, leaf_proc, boundary_keys, rank)
                #self._create_boundary_particles(&node.children[i], leaf_proc, particles, boundary_keys, rank)

    #cdef node_neighbor_search(self, Node* node, np.int32_t* leaf_proc, list particles, set boundary_keys, int rank):
    cdef node_neighbor_search(self, Node* node, ParticleContainer* part_container, np.int32_t* leaf_proc, list particles,
            set boundary_keys, int rank):
        """
        Loop over neighbor leafs of leaf, for each leaf that does not belong in the domain
        create a particle at the center of that leaf.
        """
        cdef Node *neighbor
        cdef np.int64_t neighbor_node_key
        cdef np.int32_t x, y
        cdef list node_list = list()
        cdef int i, j

        cdef set node_set = set()

        # find neighbor leaf by shifting leaf node key by half box length
        for i in range(3):
            for j in range(3):

                x = <np.int32_t> (node.center[0] + (i-1)*node.box_length)
                y = <np.int32_t> (node.center[1] + (j-1)*node.box_length)

                # exclude the leaf node from the search
                if i == j == 1:
                    continue

                # make sure the key is in the global domain
                if (self.xmin <= x and x <= self.xmax) and (self.ymin <= y and y <= self.ymax):

                    neighbor_node_key = hilbert_key_2d(x, y, self.order)

                    # find neighbor node that is at max the same level of query node
                    neighbor = self._find_node_by_key_level(neighbor_node_key, node.level)

                    # make sure we don't add duplicate neighbors
                    if neighbor.sfc_key not in node_set:

                        node_set.add(neighbor.sfc_key)

                        # check if their are sub nodes, if so collect them too
                        if neighbor.children != NULL:
                            self._subneighbor_find(neighbor, leaf_proc, particles, boundary_keys, rank, i, j)
                        else:
                            if leaf_proc[neighbor.array_index] != rank:
                                if neighbor.sfc_key not in boundary_keys:
                                    boundary_keys.add(neighbor.sfc_key)
                                    pa.make_ghost(neighbor.center[0], neighbor.center[1],
                                            leaf_proc[neighbor.array_index])
                                    #particles.append([neighbor.center[0], neighbor.center[1],
                                    #        leaf_proc[neighbor.array_index]])

                # domain bondary node
                else:
                    if (i == 1 and j != 1) or (i != 1 and j == 1):
                        pa.make_ghost(x, y, -1)
                        #particles.append([x, y, -1])

        return node_list

    cdef void _subneighbor_find(self, Node* candidate, np.int32_t* leaf_proc, list particles,
            set boundary_keys, int rank, int i, int j):
        """
        Find subneighbor leafs.
        """

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
                    self._subneighbor_find(child_cand, leaf_proc, particles, boundary_keys, rank, i, j)
                else:
                    if leaf_proc[child_cand.array_index] != rank:
                        if child_cand.sfc_key not in boundary_keys:
                            boundary_keys.add(child_cand.sfc_key)
                            particles.append([child_cand.center[0], child_cand.center[1],
                                leaf_proc[child_cand.array_index]])

    def update_hilbert_keys_and_process_id(self, np.float64_t[:] x_pos, np.float64_t[:] y_pos, np.int64_t[:] keys,
            np.int8_t[:] tags, np.int32_t[:] proc_ids, np.int32_t[:] leaf_procs, np.float64_t[:] corner,
            np.float64_t box_length, int my_proc):
        """
        real particle = 0
        boundary ghost interior = 1
        boundary ghost exterior = 2
        ghost interior = 3
        ghost exterior = 4
        """

        cdef Node *node
        cdef np.int32_t x, y
        cdef np.int64_t key

        cdef int i
        for i in xrange(x.size): # loop over all particles, real + ghost

            # map particle position into hilbert space
            x = <np.int32_t> ((x_pos[i] - corner[0])*fac)
            y = <np.int32_t> ((y_pos[i] - corner[1])*fac)

            # make sure the key is in the global domain
            if (self.xmin <= x and x <= self.xmax) and (self.ymin <= y and y <= self.ymax):

                # generate hilbert key for particle
                key = hilbert_key_2d(x, y, self.order)
                keys[i] = key

                # use key to find which leaf the particles lives in and store process id
                node = self._find_leaf(key)
                procs_id[i] = leaf_procs[node.array_index]

                # check if this particle is real or it is a interior ghost 
                if procs_id[i] != my_proc:
                    tags[i] = 3
                else:
                    tags[i] = 0

            else:

                # ghost particles outside the domain are assigned -1 key
                keys[i] = -1
                procs_id[i] = -1
                tags[i] = 4

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

    cdef void _free_nodes(self, Node* node):
        """
        Recursively release memory allocated for nodes
        """
        cdef int i
        if node.children != NULL:
            for i in range(4):
                self._free_nodes(&node.children[i])
            stdlib.free(node.children)

    def __dealloc__(self):
        """
        Release the memory allocated for the tree
        """
        if self.root != NULL:
            self._free_nodes(self.root)
            stdlib.free(self.root)
