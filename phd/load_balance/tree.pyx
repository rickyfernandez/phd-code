import numpy as np
from mpi4py import MPI
cimport libc.stdlib as stdlib

from ..utils.particle_tags import ParticleTAGS
from ..hilbert.hilbert cimport hilbert_key_2d, hilbert_key_3d

cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost
cdef int Exterior = ParticleTAGS.Exterior
cdef int Interior = ParticleTAGS.Interior

cdef class TreeMemoryPool:

    def __init__(self, int num_nodes):
        self.node_array = <Node*> stdlib.malloc(num_nodes*sizeof(Node))
        if self.node_array == NULL:
            raise MemoryError()
        self.used = 0
        self.capacity = num_nodes

    cdef Node* get(self, int count):
        """
        Allocate 'count' number of nodes from the pool and return pointer to the first node.

        Parameters
        ----------
        int : count
            Number of nodes to allocate
        """
        cdef Node* first_node
        cdef int current = self.used

        self.used += count
        if self.used >= self.capacity:
            self.resize(2*self.capacity)
        first_node = &self.node_array[current]
        return first_node

    cdef void resize(self, int size):
        """
        Resize the memory pool to have 'size' number of nodes available

        Parameters
        ----------
        int : size
            Number of nodes allocated
        """
        cdef void* node_array = NULL
        if size > self.capacity:
            node_array = <Node*> stdlib.realloc(self.node_array, size*sizeof(Node))

            if node_array == NULL:
                stdlib.free(<void*>self.node_array)
                raise MemoryError

            self.node_array = <Node*> node_array
            self.capacity = size

    cdef void reset(self):
        """
        Reset the pool
        """
        self.used = 0

    cpdef int number_leaves(self):
        """
        Return number of nodes used from the pool that are leaves.
        """
        cdef int i, num_leaves = 0
        for i in range(self.used):
            if self.node_array[i].children_start == -1:
                num_leaves += 1
        return num_leaves

    cpdef int number_nodes(self):
        """
        Return number of nodes used from the pool.
        """
        return self.used

    def __dealloc__(self):
        stdlib.free(<void*>self.node_array)


cdef class Tree:

    def __init__(self,
            np.ndarray[np.float64_t, ndim=1] corner,
            double domain_length, int dim,
            double factor, int min_in_leaf=32, int order=21):

        cdef int i

        # size of the domain is the size of hilbert space
        for i in range(dim):
            # corner of domain in physical space
            self.domain_corner[i] = corner[i]

            # hilbert space min/max
            self.bounds[0][i] = 0
            self.bounds[1][i] = 2**order
            #self.bounds[1][i] = 1 << order

        self.order = order
        self.min_in_leaf = min_in_leaf
        self.factor = factor
        self.root = NULL
        self.hilbert_func = NULL

        # original info of the particle domain
        self.domain_length = domain_length
        #self.domain_fac = (1 << order)/domain_length
        self.domain_fac = (2**order)/domain_length

        self.dim = dim
        if dim == 2:
            self.hilbert_func = hilbert_key_2d
        elif dim == 3:
            self.hilbert_func = hilbert_key_3d
        else:
            raise RuntimeError("Wrong dimension for tree")

        self.mem_pool = TreeMemoryPool(10000)

    cdef void _create_node_children(self, Node* node):
        """
        Subdivide node into children and transfer appropriate
        information to each child.

        Parameters
        ----------
        node : Node*
            Node that will be subdivided.
        """
        cdef Node* child
        cdef np.int64_t key
        cdef int child_node_index
        cdef np.float64_t center[3]
        cdef int num_children = 2**self.dim
        #cdef int num_children = 1 << self.dim

        # create children nodes
        cdef Node* new_node = self.mem_pool.get(num_children)
        node.children_start = new_node - node

        # *** should remove level? not used ***

        # pass parent data to children 
        cdef int i, k
        for i in range(num_children):

            if node.number_sfc_keys < num_children:
                raise RuntimeError("Not enough hilbert keys to be split")

            child = node + node.children_start + i

            # each child has a cut of hilbert keys from parent
            child.number_sfc_keys = node.number_sfc_keys/num_children
            child.sfc_start_key = node.sfc_start_key + i*node.number_sfc_keys/num_children
            child.particle_index_start = node.particle_index_start

            child.level = node.level + 1
            child.box_length = node.box_length/2.0

            child.number_particles = 0
            child.number_segments = 0
            child.children_start = -1

        # create children center coordinates by shifting parent
        # coordinates, children in z-order
        for i in range(num_children):
            for k in range(self.dim):
                if ((i >> k) & 1):
                    center[k] = node.center[k] + 0.25*node.box_length
                else:
                    center[k] = node.center[k] - 0.25*node.box_length

            # compute hilbert key for each child
            key = self.hilbert_func(
                    <np.int32_t> center[0],
                    <np.int32_t> center[1],
                    <np.int32_t> center[2],
                    self.order)

            # find which node this key belongs to it and store the key
            # center coordinates
            child_node_index = (key - node.sfc_start_key)/(node.number_sfc_keys/num_children)
            child = node + node.children_start + child_node_index

            # transfer spatial data to child
            child.sfc_key = key
            for k in range(self.dim):
                child.center[k] = center[k]

            # map from z-order order to hilbert
            node.zorder_to_hilbert[i] = child_node_index

#    cdef void _create_node_children(self, Node* node):
#        """
#        Subdivide node into children and transfer appropriate
#        information to each child.
#
#        Parameters
#        ----------
#        node : Node*
#            Node that will be subdivided.
#        """
#        cdef int num_children = 2**self.dim
#        #cdef int num_children = 1 << self.dim
#
#        # create children nodes
#        cdef Node* new_node = self.mem_pool.get(num_children)
#        node.children_start = new_node - node
#
#        # pass parent data to children 
#        cdef int i, j, k, m
#        for i in range(num_children):
#
#            if node.number_sfc_keys < num_children:
#                raise RuntimeError("Not enough hilbert keys to be split")
#
#            child = node + node.children_start + i
#
#            # each child has a cut of hilbert keys from parent
#            child.number_sfc_keys = node.number_sfc_keys/num_children
#            child.sfc_start_key = node.sfc_start_key + i*node.number_sfc_keys/num_children
#            child.particle_index_start = node.particle_index_start
#
#            child.level = node.level + 1
#            child.box_length = node.box_length/2.0
#
#            child.number_particles = 0
#            child.number_segments = 0
#            child.children_start = -1
#
#        # create children center coordinates by shifting parent coordinates by 
#        # half box length in each dimension
#        cdef np.int64_t key
#        cdef int child_node_index
#        for m in range(num_children):
#
#            j = 1 if m & (1 << 0) else 0
#            i = 1 if m & (1 << 1) else 0
#            k = 1 if m & (1 << 2) else 0
#
#            # compute hilbert key for each child
#            key = self.hilbert_func(
#                    <np.int32_t> (node.center[0] + (2*i-1)*node.box_length/4.0),
#                    <np.int32_t> (node.center[1] + (2*j-1)*node.box_length/4.0),
#                    <np.int32_t> (node.center[2] + (2*k-1)*node.box_length/4.0),
#                    self.order)
#
#            # find which node this key belongs to it and store the key
#            # center coordinates
#            child_node_index = (key - node.sfc_start_key)/(node.number_sfc_keys/num_children)
#            child = node + node.children_start + child_node_index
#
#            child.sfc_key = key
#            child.center[0] = node.center[0] + (2*i-1)*node.box_length/4.0
#            child.center[1] = node.center[1] + (2*j-1)*node.box_length/4.0
#            child.center[2] = node.center[2] + (2*k-1)*node.box_length/4.0
#
#            # the children are in hilbert order, this mapping allows to grab children
#            # left-back-down, left-back-up , left-front-down, left-front-up,
#            # right-back-down, right-back-up , right-front-down, right-front-up,
#            node.children_index[(i<<1) + j + (k<<2)] = child_node_index


    cpdef _build_local_tree(self, np.ndarray[np.int64_t, ndim=1] sorted_part_keys, int max_in_leaf):
        """
        Create a local tree by using hilbert keys.

        Parameters
        ----------
        np.ndarray : sorted_part_keys
            Hilbert keys for each particle used to construct local tree
        int : max_in_leaf
            Leaf splitting value
        """
        # reset tree without deallocating memory
        self.mem_pool.reset()
        self.root = self.mem_pool.get(1)

        # create root node which holds all possible hilbert
        # keys in a grid of 2^order resolution per dimension
        self.root.children_start = -1
        self.root.sfc_start_key = 0
        self.root.number_sfc_keys = 2**(self.dim*self.order)
        #self.root.number_sfc_keys = 1 << (self.dim*self.order)
        self.root.particle_index_start = 0
        self.root.level = 0
        self.root.box_length = 2**self.order
        #self.root.box_length = 1 << self.order
        self.root.number_particles = sorted_part_keys.size
        self.root.number_segments = 0 # not used for local tree

        for i in range(self.dim):
            # the center of the root is the center of the grid of hilbert space
            self.root.center[i] = 0.5*self.root.box_length

        # create the local tree
        self._fill_particles_nodes(self.root, &sorted_part_keys[0], max_in_leaf)

    cdef void _fill_particles_nodes(self, Node* node, np.int64_t* sorted_part_keys, int max_in_leaf):
        """
        Subdivide node and bin each particle to the appropriate child. This function is recrusive
        and will continue untill each child contains less than max_in_leaf particles.

        Parameters
        ----------
        node : Node
            Node that will be subdivided.
        sorted_part_keys : np.int64_t*
           Pointer to sorted hilbert keys of particles
        max_in_leaf : int
            Max number of particles in a node.
        """
        cdef Node* child
        cdef int i, child_node_index
        cdef int num_children = 2**self.dim
        #cdef int num_children = 1 << self.dim

        self._create_node_children(node)

        # loop over parent particles and assign them to proper child
        for i in range(node.particle_index_start, node.particle_index_start + node.number_particles):

            # which child node does this particle belong to
            child_node_index = (sorted_part_keys[i] - node.sfc_start_key)/(node.number_sfc_keys/num_children)

            if child_node_index < 0 or child_node_index > num_children:
                raise RuntimeError("hilbert key out of bounds")

            child = node + node.children_start + child_node_index

            # if child node is empty then this particle is first in cut
            if child.number_particles == 0:
                child.particle_index_start = i

            # update number of particles of child
            child.number_particles += 1

        # if child has more particles then the maximum allowed, then subdivide 
        for i in range(num_children):
            child = node + node.children_start + i
            if child.number_particles > max_in_leaf:
                self._fill_particles_nodes(child, sorted_part_keys, max_in_leaf)

    cdef void _build_global_tree(self, int global_num_particles, np.ndarray[np.int64_t, ndim=1] sorted_segm_keys,
            np.ndarray[np.int32_t, ndim=1] sorted_segm_parts, int max_in_leaf):
        """
        Create a global tree by using hilbert segments for local tree construction.

        Parameters
        ----------
        int : global_num_particles
            Total number of particles across all processors
        np.ndarray : sorted_part_keys
            Starting hilbert key for each segment
        np.ndarray : sorted_segm_parts
            Number of particles in each segment
        """
        # reset tree without deallocating memory
        self.mem_pool.reset()
        self.root = self.mem_pool.get(1)

        # create root node which holds all possible hilbert
        # keys in a grid of 2^order resolution per dimension
        self.root.children_start = -1
        self.root.sfc_start_key = 0
        self.root.number_sfc_keys = 2**(self.dim*self.order)
        #self.root.number_sfc_keys = 1 << (self.dim*self.order)
        self.root.particle_index_start = 0
        self.root.level = 0
        self.root.box_length = 2**self.order
        #self.root.box_length = 1 << self.order
        self.root.number_particles = global_num_particles
        self.root.number_segments = sorted_segm_keys.size

        for i in range(self.dim):
            # the center of the root is the center of the grid of hilbert space
            self.root.center[i] = 0.5*self.root.box_length

        # create the local tree
        self._fill_segments_nodes(self.root, &sorted_segm_keys[0], &sorted_segm_parts[0], max_in_leaf)

    cdef void _fill_segments_nodes(self, Node* node, np.int64_t* sorted_segm_keys, np.int32_t* sorted_segm_parts, int max_in_leaf):
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
        cdef int num_children = 2**self.dim
        #cdef int num_children = 1 << self.dim
        cdef int i, child_node_index

        self._create_node_children(node)

        # loop over parent segments and assign them to proper child
        for i in range(node.particle_index_start, node.particle_index_start + node.number_segments):

            # which node does this segment belong to
            child_node_index = (sorted_segm_keys[i] - node.sfc_start_key)/(node.number_sfc_keys/num_children)

            if child_node_index < 0 or child_node_index > num_children:
                raise RuntimeError("hilbert key out of bounds")

            child = node + node.children_start + child_node_index

            if child.number_segments == 0:
                child.particle_index_start = i

            # update the number of particles for child
            child.number_particles += sorted_segm_parts[i]
            child.number_segments += 1

        # block to ensure tree does not overly subdivide
        for i in range(num_children):
            child = node + node.children_start + i
            if child.number_particles < self.min_in_leaf:
                node.children_start = -1            # set parent to leaf
                self.mem_pool.used -= num_children  # release memory
                return

        # if child has more particles then the maximum allowed, then subdivide 
        for i in range(num_children):
            child = node + node.children_start + i
            if child.number_particles > max_in_leaf:
                self._fill_segments_nodes(child, sorted_segm_keys, sorted_segm_parts, max_in_leaf)

    cdef void construct_global_tree(self, CarrayContainer pc, object comm):
        """
        Create global tree on all processors.

        Parameters
        ----------
        pc : CarrayContainer
            Carray container holding all particle information
        comm : object
            Parallel class responsible for all communications
        """
        cdef np.ndarray ind
        cdef np.ndarray arg, offsets
        cdef np.ndarray sendbuf, recvbuf
        cdef np.ndarray keys, sorted_keys
        cdef np.ndarray leaf_keys, leaf_part
        cdef np.ndarray glb_leaf_keys, glb_leaf_part

        cdef int i, j, size
        cdef int max_in_leaf
        cdef int glb_num_part
        cdef int glb_num_leaves, lc_num_leaves, num_leaves

        size = comm.Get_size()

        # collect number of particles from all process
        sendbuf = np.array([pc.get_carray_size()], dtype=np.int32)
        recvbuf = np.empty(size, dtype=np.int32)

        comm.Allgather(sendbuf=sendbuf, recvbuf=recvbuf)
        glb_num_part = np.sum(recvbuf)

        # sort hilbert keys
        keys = pc["key"]
        sorted_keys = np.sort(keys)

        # create local tree first
        max_in_leaf = <int> (self.factor*glb_num_part/(size**2))
        #max_in_leaf = <int> (self.factor*glb_num_part/(size<<2))
        self._build_local_tree(sorted_keys, max_in_leaf)

        # count local number of leaves
        lc_num_leaves = self.mem_pool.number_leaves()

        # collect leaves from all process  
        leaf_keys = np.empty(lc_num_leaves, dtype=np.int64)
        leaf_part = np.empty(lc_num_leaves, dtype=np.int32)

        # leaf information is just key and number of particles in it
        j = 0
        for i in range(self.mem_pool.used):
            if self.mem_pool.node_array[i].children_start == -1:
                leaf_keys[j] = self.mem_pool.node_array[i].sfc_start_key
                leaf_part[j] = self.mem_pool.node_array[i].number_particles
                j += 1

        # prepare to bring all leafs form all local trees
        sendbuf[0] = lc_num_leaves
        comm.Allgather(sendbuf=sendbuf, recvbuf=recvbuf)

        glb_num_leaves = np.sum(recvbuf)
        offsets = np.zeros(size)
        offsets[1:] = np.cumsum(recvbuf)[:-1]

        glb_leaf_keys = np.empty(glb_num_leaves, dtype=np.int64)
        glb_leaf_part = np.empty(glb_num_leaves, dtype=np.int32)

        comm.Allgatherv(leaf_keys, [glb_leaf_keys, recvbuf, offsets, MPI.INT64_T])
        comm.Allgatherv(leaf_part, [glb_leaf_part, recvbuf, offsets, MPI.INT])

        # sort global leaves
        ind = glb_leaf_keys.argsort()
        glb_leaf_keys[:] = glb_leaf_keys[ind]
        glb_leaf_part[:] = glb_leaf_part[ind]

        # rebuild tree using local leaves from alll processors
        max_in_leaf = <int> (self.factor*glb_num_part/size)
        self._build_global_tree(glb_num_part, glb_leaf_keys, glb_leaf_part, max_in_leaf)

        # assign global leaves to array order
        num_leaves = 0
        self._leaves_to_array(self.root, &num_leaves)
        self.number_leaves = num_leaves

    cdef void _leaves_to_array(self, Node* node, int* num_leaves):
        """
        Recursively walk the tree mapping each leaf to an array index.

        Paramters
        --------
        node : Node
        """
        cdef int i
        if node.children_start == -1:
            node.array_index = num_leaves[0]
            num_leaves[0] += 1
        else:
            for i in range(1 << self.dim):
                self._leaves_to_array(node + node.children_start + i, num_leaves)

    cdef Node* find_leaf(self, np.int64_t key):
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

    cdef int get_nearest_process_neighbors(self, double center[3], double h,
            LongArray leaf_pid, int rank, LongArray nbrs):
        """
        Gather all processors enclosed in square.

        Parameters
        ----------
        center : double array
            Center of search square in physical space.
        h : double
            Box length of search square in physical space.
        leaf_pid : LongArray
            Array of processors ids, index corresponds to a leaf in global tree.
        rank : int
            Current processor.
        nbrs : LongArray
            Container to hold all the processors ids.
        """
        cdef double smin[3]
        cdef double smax[3]
        cdef int i, j

        # scale to hilbert coordinates
        for i in range(self.dim):
            smin[i] = ((center[i] - h) - self.domain_corner[i])*self.domain_fac
            smax[i] = ((center[i] + h) - self.domain_corner[i])*self.domain_fac

        self._neighbors(self.root, smin, smax, leaf_pid.data, rank, nbrs)

        # remove duplicates
#        qsort(<void*> &self.nbrs.data[0], <size_t> self.nbrs.length,
#                sizeof(int), int_compare)
#        if nbrs.length != 0:
#            j = 0
#            for i in range(1, nbrs.length):
#                if nbrs.data[i] != nbrs.data[j]:
#                    j += 1
#                    nbrs.data[j] = nbrs.data[i]
#            nbrs.resize(j + 1)

        # return number of processors found
        return nbrs.length

    cdef void _neighbors(self, Node* node, double smin[3], double smax[3], np.int32_t* leaf_pid, int rank, LongArray nbrs):
        cdef int i

        # is node a leaf
        if node.children_start == -1:
            # dont count nodes on our process
            if rank != leaf_pid[node.array_index]:
                nbrs.append(leaf_pid[node.array_index])

        else:
            # does search box overlap with node
            for i in range(self.dim):
                if (node.center[i] + 0.5*node.box_length) < smin[i]: return
                if (node.center[i] - 0.5*node.box_length) > smax[i]: return

            # node overlaps open sub nodes
            for i in range(1 << self.dim):
                self._neighbors(node + node.children_start + i, smin, smax, leaf_pid, rank, nbrs)

    # temporary function to do outputs in python
    def dump_data(self):
        cdef list data_list = []
        cdef Node node

        cdef int i
        for i in range(self.mem_pool.used):
            node = self.mem_pool.node_array[i]
            if node.children_start == -1:
                data_list.append([
                    node.center[0],
                    node.center[1],
                    node.center[2],
                    node.box_length,
                    node.number_particles])

        return data_list
