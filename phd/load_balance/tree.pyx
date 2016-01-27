import numpy as np

cimport cython
cimport numpy as np
#from libc.posix cimport off_t
cimport libc.stdlib as stdlib
from containers.containers cimport ParticleContainer, CarrayContainer
from utils.carray cimport BaseArray, DoubleArray, IntArray, LongArray, LongLongArray

from hilbert.hilbert cimport hilbert_key_2d, hilbert_key_3d
from utils.particle_tags import ParticleTAGS

cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost
cdef int Exterior = ParticleTAGS.Exterior
cdef int Interior = ParticleTAGS.Interior
cdef int ExportInterior = ParticleTAGS.ExportInterior
cdef int OldGhost = ParticleTAGS.OldGhost

cdef class TreeMemoryPool:

    def __init__(self, int num_nodes):
        self.node_array = <Node*> stdlib.malloc(num_nodes*sizeof(Node))
        if self.node_array == <Node*> NULL:
            raise MemoryError
        self.used = 0
        self.capacity = num_nodes

    cdef Node* get(self, int count):
        cdef Node* first_node = &self.node_array[self.used]
        self.used += count
        if self.used >= self.capacity:
            raise MemoryError("what the hell")
        return first_node

    cdef void resize(self, int size):
        cdef void* node_array = NULL
        if size > self.capacity:
            node_array = <Node*> stdlib.realloc(self.node_array, size*sizeof(Node))

            if node_array == NULL:
                stdlib.free(<void*> self.node_array)
                raise MemoryError

            self.node_array = <Node*> node_array
            self.capacity = size

    cdef void reset(self):
        self.used = 0

    def __dealloc_(self):
        stdlib.free(<void*>self.node_array)

cdef class TreeBase:

    def __init__(self, int total_num_part,
            np.ndarray[np.int64_t, ndim=1] sorted_part_keys,
            np.ndarray[np.float64_t, ndim=1] corner,
            double domain_length,
            np.ndarray[np.int64_t, ndim=1] sorted_segm_keys=None,
            np.ndarray[np.int32_t, ndim=1] num_part_leaf=None,
            int total_num_process=1, double factor=1.0, int order=21):

        self.order = order
        self.factor = factor
        self.sorted_part_keys = sorted_part_keys
        self.total_num_part = total_num_part
        self.total_num_process = total_num_process
        self.root = NULL
        self.hilbert_func = NULL

        # original info of the particle domain
        self.domain_corner = corner
        self.domain_length = domain_length
        self.domain_fac = (1 << order)/domain_length

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
        if node.children_start == -1:
            node.array_index = self.number_leaves
            self.number_leaves += 1
        else:
            for i in range(1 << self.dim):
                self._assign_leaves_to_array(node + node.children_start + i)

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

        self.root = self.mem_pool.get(1)

        # create root node which holds all possible hilbert
        # keys in a grid of 2^order resolution per dimension
        self.root.children_start = -1
        self.root.sfc_start_key = 0
        self.root.number_sfc_keys = 2**(self.dim*self.order)
        self.root.particle_index_start = 0
        self.root.level = 0
        self.root.box_length = 2**(self.order)

        cdef int i
        for i in range(1 << self.dim):
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
        for i in range(keys.shape[0]):
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
        for i in range(keys.shape[0]):
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
        for i in range(keys.shape[0]):

            node = self._find_leaf(keys[i])
            proc = leaf_procs[node.array_index]

            if proc != my_proc:
                part_ids[count] = i
                proc_ids[count] = proc
                count += 1

    cdef void _create_node_children(self, Node* node):
        """
        Subdivide node into children and transfer appropriate
        information to each child.

        Parameters
        ----------
        node : Node*
            Node that will be subdivided.
        """
        cdef int num_children = 1 << self.dim

        # create children nodes
        cdef Node* new_node = self.mem_pool.get(num_children)
        node.children_start = new_node - node

        # pass parent data to children 
        cdef int i, j, k, m
        for i in range(num_children):

            if node.number_sfc_keys < num_children:
                raise RuntimeError("Not enough hilbert keys to be split")

            child = (node + node.children_start + i)

            # each child has a cut of hilbert keys from parent
            child.number_sfc_keys = node.number_sfc_keys/num_children
            child.sfc_start_key   = node.sfc_start_key + i*node.number_sfc_keys/num_children
            child.particle_index_start = node.particle_index_start

            child.level = node.level + 1
            child.box_length = node.box_length/2.0

            child.number_particles = 0
            child.number_segments = 0
            child.children_start = -1

        # create children center coordinates by shifting parent coordinates by 
        # half box length in each dimension
        cdef np.int64_t key
        cdef int child_node_index
        for m in range(num_children):

            j = 1 if m & (1 << 0) else 0
            i = 1 if m & (1 << 1) else 0
            k = 1 if m & (1 << 2) else 0

            # compute hilbert key for each child
            key = self.hilbert_func(
                    <np.int32_t> (node.center[0] + (2*i-1)*node.box_length/4.0),
                    <np.int32_t> (node.center[1] + (2*j-1)*node.box_length/4.0),
                    <np.int32_t> (node.center[2] + (2*k-1)*node.box_length/4.0),
                    self.order)

            # find which node this key belongs to it and store the key
            # center coordinates
            child_node_index = (key - node.sfc_start_key)/(node.number_sfc_keys/num_children)
            child = (node + node.children_start + child_node_index)

            child.sfc_key = key
            child.center[0] = node.center[0] + (2*i-1)*node.box_length/4.0
            child.center[1] = node.center[1] + (2*j-1)*node.box_length/4.0
            child.center[2] = node.center[2] + (2*k-1)*node.box_length/4.0

            # the children are in hilbert order, this mapping allows to grab children
            # left-back-down, left-back-up , left-front-down, left-front-up,
            # right-back-down, right-back-up , right-front-down, right-front-up,
            node.children_index[(i<<1) + j + (k<<2)] = child_node_index

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
        cdef Node* child
        cdef num_children = 1 << self.dim
        cdef int i, child_node_index

        self._create_node_children(node)

        # loop over parent particles and assign them to proper child
        for i in range(node.particle_index_start, node.particle_index_start + node.number_particles):

            # which node does this particle/segment belong to
            child_node_index = (self.sorted_part_keys[i] - node.sfc_start_key)/(node.number_sfc_keys/num_children)

            if child_node_index < 0 or child_node_index > num_children:
                raise RuntimeError("hilbert key out of bounds")

            child = node + node.children_start + child_node_index

            # if child node is empty then this is the first particle in the cut
            if child.number_particles == 0:
                child.particle_index_start = i

            # update the number of particles for child
            child.number_particles += 1

        # if child has more particles then the maximum allowed, then subdivide 
        for i in range(num_children):
            child = node + node.children_start + i
            if child.number_particles > max_in_leaf:
                self._fill_particles_nodes(child, max_in_leaf)

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
        cdef int num_children = 1 << self.dim
        cdef int i, child_node_index

        self._create_node_children(node)

        # loop over parent segments and assign them to proper child
        for i in range(node.particle_index_start, node.particle_index_start + node.number_segments):

            # which node does this segment belong to
            child_node_index = (self.sorted_segm_keys[i] - node.sfc_start_key)/(node.number_sfc_keys/num_children)

            if child_node_index < 0 or child_node_index > num_children:
                raise RuntimeError("hilbert key out of bounds")

            child = (node + node.children_start + child_node_index)

            if child.number_segments == 0:
                child.particle_index_start = i

            # update the number of particles for child
            child.number_particles += self.num_part_leaf[i]
            child.number_segments += 1

        # if child has more particles then the maximum allowed, then subdivide 
        for i in range(num_children):
            child = node + node.children_start + i
            if child.number_particles > max_in_leaf:
                self._fill_segments_nodes(child, max_in_leaf)

    cdef void _count_leaves(self, Node* node):
        """
        Recursively count the number of leaves by walking the tree.

        Parameters
        ----------
        node : Node
        """
        cdef int i
        if node.children_start == -1:
            self.number_leaves += 1
        else:
            for i in range(1 << self.dim):
                self._count_leaves(node + node.children_start + i)

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
        return self.mem_pool.used

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
        if node.children_start == -1:

            start_keys[counter[0]] = node.sfc_start_key
            num_part_leaf[counter[0]] = node.number_particles
            counter[0] += 1

        else:
            for i in range(1 << self.dim):
                self._collect_leaves_for_export(node + node.children_start + i,
                        start_keys, num_part_leaf, counter)

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
        cdef int num_children = 1 << self.dim

        node = self.root
        while node.children_start != -1:
            child_node_index = (key - node.sfc_start_key)/(node.number_sfc_keys/num_children)
            node = node + node.children_start + child_node_index

        return node

#    cdef Node* _find_node_by_key_level(self, np.uint64_t key, np.uint32_t level):
#        """
#        Find node that contains given hilbert key. The node can be at most *level* down
#        the tree.
#
#        Parameters
#        ----------
#        key : int64
#            Hilbert key used for search.
#        int : int
#            The max depth the node can be in the tree
#
#        Returns
#        -------
#        node : Node
#            Node that contains hilbert key.
#        """
#        cdef Node* candidate = self.root
#        cdef int child_node_index
#
#        while candidate.level < level and candidate.children != NULL:
#            child_node_index = (key - candidate.sfc_start_key)/(candidate.number_sfc_keys/4)
#            candidate = &candidate.children[child_node_index]
#
#        return candidate
#
#    def create_boundary_particles(self, ParticleContainer part_array, int rank, np.int32_t[:] leaf_proc):
#        """create boundary ghost particles"""
#        cdef set boundary_keys = set()
#        self._create_boundary_particles(self.root, part_array, &leaf_proc[0], boundary_keys, rank)
#
#    cdef void _create_boundary_particles(self, Node* node, ParticleContainer part_array, np.int32_t* leaf_proc,
#            set boundary_keys, int rank):
#        cdef int i
#        if node.children == NULL:
#            # leaf belongs to our domain
#            if leaf_proc[node.array_index] == rank:
#                self.node_neighbor_search(node, part_array, leaf_proc, boundary_keys, rank)
#        else:
#            for i in range(4):
#                self._create_boundary_particles(&node.children[i], part_array, leaf_proc, boundary_keys, rank)
#
#    cdef void node_neighbor_search(self, Node* node, ParticleContainer part_array, np.int32_t* leaf_proc,
#            set boundary_keys, int rank):
#        """
#        Loop over neighbor leafs of leaf, for each leaf that does not belong in the domain
#        create a particle at the center of that leaf.
#        """
#        cdef Node *neighbor
#        cdef np.int64_t neighbor_node_key
#        cdef np.int32_t x, y
#        cdef list node_list = list()
#        cdef int i, j
#
#        cdef set node_set = set()
#
#        # find neighbor leaf by shifting leaf node key by half box length
#        for i in range(3):
#            for j in range(3):
#
#                x = <np.int32_t> (node.center[0] + (i-1)*node.box_length)
#                y = <np.int32_t> (node.center[1] + (j-1)*node.box_length)
#
#                # exclude the leaf node from the search
#                if i == j == 1:
#                    continue
#
#                # make sure the key is in the global domain
#                if (self.xmin <= x and x <= self.xmax) and (self.ymin <= y and y <= self.ymax):
#
#                    neighbor_node_key = hilbert_key_2d(x, y, self.order)
#
#                    # find neighbor node that is at max the same level of query node
#                    neighbor = self._find_node_by_key_level(neighbor_node_key, node.level)
#
#                    # make sure we don't add duplicate neighbors
#                    if neighbor.sfc_key not in node_set:
#
#                        node_set.add(neighbor.sfc_key)
#
#                        # check if their are sub nodes, if so collect them too
#                        if neighbor.children != NULL:
#                            self._subneighbor_find(neighbor, leaf_proc, part_array, boundary_keys, rank, i, j)
#                        else:
#                            if leaf_proc[neighbor.array_index] != rank:
#                                if neighbor.sfc_key not in boundary_keys:
#                                    boundary_keys.add(neighbor.sfc_key)
#                                    part_array.make_ghost(neighbor.center[0]/self.domain_fac + self.domain_corner[0],
#                                            neighbor.center[1]/self.domain_fac + self.domain_corner[1],
#                                            leaf_proc[neighbor.array_index])
#                # we have bondary node
#                else:
#
#                    # non corner local domain node
#                    if (i == 1 and j != 1) or (i != 1 and j == 1):
#                        part_array.make_ghost((<np.float64_t> x)/self.domain_fac + self.domain_corner[0],
#                                (<np.float64_t> y)/self.domain_fac + self.domain_corner[1], -1)
#
#
#                    # corner local domain node left/right
#                    elif (x <= self.xmin or self.xmax <= x) and (self.ymin <= y and y <= self.ymax):
#
#                        neighbor_node_key = hilbert_key_2d( <np.int32_t> node.center[0], y, self.order)
#                        neighbor = self._find_leaf(neighbor_node_key)
#                        if leaf_proc[neighbor.array_index] != rank:
#                            part_array.make_ghost((<np.float64_t> x)/self.domain_fac + self.domain_corner[0],
#                                    (<np.float64_t> y)/self.domain_fac + self.domain_corner[1], -1)
#
#                    # corner local domain node above/below
#                    elif (y <= self.ymin or self.ymax <= y) and (self.xmin <= x and x <= self.xmax):
#
#                        neighbor_node_key = hilbert_key_2d(x, <np.int32_t> node.center[1], self.order)
#                        neighbor = self._find_leaf(neighbor_node_key)
#                        if leaf_proc[neighbor.array_index] != rank:
#                            part_array.make_ghost((<np.float64_t> x)/self.domain_fac + self.domain_corner[0],
#                                    (<np.float64_t> y)/self.domain_fac + self.domain_corner[1], -1)
#
#                    # corner global domain node
#                    else:
#                        part_array.make_ghost((<np.float64_t> x)/self.domain_fac + self.domain_corner[0],
#                                (<np.float64_t> y)/self.domain_fac + self.domain_corner[1], -1)
#
#
#    cdef void _subneighbor_find(self, Node* candidate, np.int32_t* leaf_proc, ParticleContainer part_array,
#            set boundary_keys, int rank, int i, int j):
#        """
#        Find subneighbor leafs.
#        """
#
#        if i == j == 1: return
#
#        cdef Node* child_cand
#        cdef np.int64_t num_loop[2], index[2], off[2][2], ii, ij
#
#        index[0] = i
#        index[1] = j
#
#        # num_steps and walk?
#        for ii in range(2):
#
#            # no offset 
#            if index[ii] == 1:
#                num_loop[ii] = 2
#                off[ii][0] = 0
#                off[ii][1] = 1
#
#            # left offset
#            elif index[ii] == 0:
#                num_loop[ii] = 1
#                off[ii][0] = 1
#
#            # right offset
#            elif index[ii] == 2:
#                num_loop[ii] = 1
#                off[ii][0] = 0
#
#        for ii in range(num_loop[0]):
#            for ij in range(num_loop[1]):
#
#                child_index = (off[0][ii] << 1) + off[1][ij]
#                child_cand = &candidate.children[candidate.children_index[child_index]]
#
#                if child_cand.children != NULL:
#                    self._subneighbor_find(child_cand, leaf_proc, part_array, boundary_keys, rank, i, j)
#                else:
#                    if leaf_proc[child_cand.array_index] != rank:
#                        if child_cand.sfc_key not in boundary_keys:
#                            boundary_keys.add(child_cand.sfc_key)
#                            part_array.make_ghost(child_cand.center[0]/self.domain_fac + self.domain_corner[0],
#                                    child_cand.center[1]/self.domain_fac + self.domain_corner[1],
#                                    leaf_proc[child_cand.array_index])
#
#    def update_particle_process(self, CarrayContainer pa, int rank, np.int32_t[:] leaf_procs):
#
#        cdef IntArray tags  = pa.get_carray("tag")
#        cdef LongArray proc = pa.get_carray("process")
#        cdef DoubleArray x  = pa.get_carray("position-x")
#        cdef DoubleArray y  = pa.get_carray("position-y")
#
#        cdef Node *node
#        cdef np.int32_t xh, yh
#        cdef np.int64_t key
#
#        cdef int i
#        for i in range(pa.get_number_of_items()):
#
#            # map particle position into hilbert space
#            xh = <np.int32_t> ((x.data[i] - self.domain_corner[0])*self.domain_fac)
#            yh = <np.int32_t> ((y.data[i] - self.domain_corner[1])*self.domain_fac)
#
#            # make sure the key is in the global domain
#            if (self.xmin <= xh and xh <= self.xmax) and (self.ymin <= yh and yh <= self.ymax):
#
#                # generate hilbert key for particle
#                key = hilbert_key_2d(xh, yh, self.order)
#
#                # use key to find which leaf the particles lives in and store process id
#                node = self._find_leaf(key)
#                proc.data[i] = leaf_procs[node.array_index]
#
#            else:
#
#                proc.data[i] = -1
#
#
#    def update_particle_domain_info(self, ParticleContainer pc, int my_proc, np.int32_t[:] leaf_procs):
#
#        cdef IntArray tags = pc.get_carray("tag")
#        cdef IntArray type = pc.get_carray("type")
#
#        cdef DoubleArray x = pc.get_carray("position-x")
#        cdef DoubleArray y = pc.get_carray("position-y")
#
#        cdef LongLongArray keys = pc.get_carray("key")
#        cdef LongArray proc = pc.get_carray("process")
#
#        cdef Node *node
#        cdef np.int32_t xh, yh
#        cdef np.int64_t key
#
#        cdef int i, npart = pc.get_number_of_particles()
#        for i in range(npart):
#
#            # map particle position into hilbert space
#            xh = <np.int32_t> ((x.data[i] - self.domain_corner[0])*self.domain_fac)
#            yh = <np.int32_t> ((y.data[i] - self.domain_corner[1])*self.domain_fac)
#
#            # make sure the key is in the global domain
#            if (self.xmin <= xh and xh <= self.xmax) and (self.ymin <= yh and yh <= self.ymax):
#
#                # generate hilbert key for particle
#                key = hilbert_key_2d(xh, yh, self.order)
#                keys.data[i] = key
#
#                # use key to find which leaf the particles lives in and store process id
#                node    = self._find_leaf(key)
#                proc_id = leaf_procs[node.array_index]
#
#                # particle is a real particle
#                if proc_id == my_proc:
#                    proc.data[i] = proc_id
#                    tags.data[i] = Real
#                    type.data[i] = Real
#
#                else:
#                    proc.data[i] = proc_id
#                    tags.data[i] = OldGhost
#                    type.data[i] = Interior
#
#            else:
#                # ghost particles outside the domain
#                proc.data[i] = -1
#                tags.data[i] = OldGhost
#                type.data[i] = Exterior
#
#    def flag_migrate_particles(self, ParticleContainer pc, int my_proc, np.int32_t[:] leaf_procs):
#
#        cdef IntArray tags = pc.get_carray("tag")
#        cdef IntArray type = pc.get_carray("type")
#
#        cdef DoubleArray x = pc.get_carray("position-x")
#        cdef DoubleArray y = pc.get_carray("position-y")
#
#        cdef LongLongArray keys = pc.get_carray("key")
#        cdef LongArray proc = pc.get_carray("process")
#
#        cdef Node *node
#        cdef np.int32_t xh, yh
#        cdef np.int64_t key
#
#        cdef int i, npart = pc.get_number_of_particles()
#        for i in range(npart):
#
#            # map particle position into hilbert space
#            xh = <np.int32_t> ((x.data[i] - self.domain_corner[0])*self.domain_fac)
#            yh = <np.int32_t> ((y.data[i] - self.domain_corner[1])*self.domain_fac)
#
#            # make sure the key is in the global domain
#            if (self.xmin <= xh and xh <= self.xmax) and (self.ymin <= yh and yh <= self.ymax):
#
#                # generate hilbert key for particle
#                key = hilbert_key_2d(xh, yh, self.order)
#                keys.data[i] = key
#
#                # use key to find which leaf the particles lives in and store process id
#                node    = self._find_leaf(key)
#                proc_id = leaf_procs[node.array_index]
#
#                # particle was real at previous time step
#                if proc.data[i] == my_proc:
#
#                    # real particle remains in domain
#                    if proc_id == my_proc:
#                        tags.data[i] = Real
#                        type.data[i] = Real
#
#                    # real particle left patch must be exported
#                    else:
#                        proc.data[i] = proc_id
#                        tags.data[i] = OldGhost
#                        type.data[i] = ExportInterior
#
#                # previously ghost and will remain ghost
#                else:
#                    tags.data[i] = OldGhost
#                    type.data[i] = Interior
#
#            else:
#
#                # ghost particles outside the domain, store it
#                tags.data[i] = OldGhost
#                type.data[i] = Exterior
#
#    # temporary function to do outputs in python
    def dump_data(self):
        cdef list data_list = []
        cdef Node node

        cdef int i
        for i in range(self.mem_pool.used):
            node = self.mem_pool.node_array[i]
            data_list.append([node.center[0],
                node.center[1], node.box_length,
                node.level,
                node.particle_index_start,
                node.number_particles])

        return data_list

cdef class QuadTree(TreeBase):

    def __init__(self, int total_num_part,
            np.ndarray[np.int64_t, ndim=1] sorted_part_keys,
            np.ndarray[np.float64_t, ndim=1] corner,
            double domain_length,
            np.ndarray[np.int64_t, ndim=1] sorted_segm_keys=None,
            np.ndarray[np.int32_t, ndim=1] num_part_leaf=None,
            int total_num_process=1, double factor=1.0, int order=21):

        TreeBase.__init__(self,
                total_num_part,
                sorted_part_keys,
                corner,
                domain_length,
                sorted_segm_keys,
                num_part_leaf,
                total_num_process,
                factor,
                order)

        self.dim = 2
        self.hilbert_func = hilbert_key_2d

        # size of the domain is the size of hilbert space
        self.xmin = self.ymin = 0
        self.xmax = self.ymax = 2**(order)

        self.mem_pool = TreeMemoryPool(100)

cdef class OcTree(TreeBase):

    def __init__(self, int total_num_part,
            np.ndarray[np.int64_t, ndim=1] sorted_part_keys,
            np.ndarray[np.float64_t, ndim=1] corner,
            double domain_length,
            np.ndarray[np.int64_t, ndim=1] sorted_segm_keys=None,
            np.ndarray[np.int32_t, ndim=1] num_part_leaf=None,
            int total_num_process=1, double factor=1.0, int order=21):

        TreeBase.__init__(self,
                total_num_part,
                sorted_part_keys,
                corner,
                domain_length,
                sorted_segm_keys,
                num_part_leaf,
                total_num_process,
                factor,
                order)

        self.dim = 3
        self.hilbert_func = hilbert_key_3d

        # size of the domain is the size of hilbert space
        self.xmin = self.ymin = self.zmin =  0
        self.xmax = self.ymax = self.zmax = 2**(order)

        self.mem_pool = TreeMemoryPool(100)
