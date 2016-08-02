import numpy as np

cimport cython
cimport numpy as np
cimport libc.stdlib as stdlib

from utils.particle_tags import ParticleTAGS
from hilbert.hilbert cimport hilbert_key_2d, hilbert_key_3d
from containers.containers cimport ParticleContainer, CarrayContainer
from utils.carray cimport BaseArray, DoubleArray, IntArray, LongArray, LongLongArray

cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost
cdef int Exterior = ParticleTAGS.Exterior
cdef int Interior = ParticleTAGS.Interior

cdef class TreeMemoryPool:

    def __init__(self, int num_nodes):
        self.node_array = <Node*> stdlib.malloc(num_nodes*sizeof(Node))
        if self.node_array == <Node*> NULL:
            raise MemoryError
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
        cdef Node* first_node = &self.node_array[current]
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
                stdlib.free(<void*> self.node_array)
                raise MemoryError

            self.node_array = <Node*> node_array
            self.capacity = size

    cpdef int number_leaves(self):
        """
        Return number of nodes used from the pool that are leaves.
        """
        cdef int i, num_leaves = 0
        for i in range(self.used):
            self.node_array[i].children_start == -1:
                num_leaves += 1
        return num_leaves

    cpdef int number_nodes(self):
        """
        Return number of nodes used from the pool.
        """
        return self.used

    cdef void reset(self):
        """
        Reset the pool
        """
        self.used = 0

    def __dealloc_(self):
        stdlib.free(<void*>self.node_array)

cdef class BaseTree:

    def __init__(self,
            np.ndarray[np.float64_t, ndim=1] corner,
            double domain_length,
            int total_num_proc=1, double factor=1.0, int min_in_leaf=32, int order=21):

        self.total_num_proc = total_num_proc
        self.order = order
        self.min_in_leaf = min_in_leaf
        self.factor = factor
        self.root = NULL
        self.hilbert_func = NULL

        # original info of the particle domain
        self.domain_corner = corner
        self.domain_length = domain_length
        self.domain_fac = (1 << order)/domain_length

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
            child.sfc_start_key = node.sfc_start_key + i*node.number_sfc_keys/num_children
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

    cdef _build_local_tree(self, np.ndarray[np.int64_t, ndim=1] sorted_part_keys, int max_in_leaf):
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
        self.root.number_sfc_keys = pow(2, self.dim*self.order)
        self.root.particle_index_start = 0
        self.root.level = 0
        self.root.box_length = pow(2, self.order)
        self.root.number_particles = sorted_part_keys.shape[0]
        self.root.number_segments = 0 # not used for local tree

        for i in range(self.dim):
            # the center of the root is the center of the grid of hilbert space
            self.root.center[i] = 0.5*self.root.box_length

        # create the local tree
        self._fill_particles_nodes(self.root, &sorted_part_keys[0], max_leaf)

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
        cdef int num_children = 1 << self.dim

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

    cdef _build_global_tree(self, int global_num_particles, np.ndarray[np.int64_t, ndim=1] sorted_segm_keys):
        """
        Create a local tree by using hilbert keys.

        Parameters
        ----------
        np.ndarray : sorted_part_keys
            Hilbert keys for each particle used to construct local tree
        """
        # reset tree without deallocating memory
        self.mem_pool.reset()
        self.root = self.mem_pool.get(1)

        # create root node which holds all possible hilbert
        # keys in a grid of 2^order resolution per dimension
        self.root.children_start = -1
        self.root.sfc_start_key = 0
        self.root.number_sfc_keys = pow(2, self.dim*self.order)
        self.root.particle_index_start = 0
        self.root.level = 0
        self.root.box_length = pow(2, self.order)
        self.root.number_particles = global_num_particles
        self.root.number_segments = sorted_segm_keys.size

        for i in range(self.dim):
            # the center of the root is the center of the grid of hilbert space
            self.root.center[i] = 0.5*self.root.box_length

        # create the local tree
        self._fill_segments_nodes(self.root, &sorted_segm_keys[0], max_leaf)

    cdef void _fill_segments_nodes(self, Node* node, np.float64_t* sorted_segm_keys, int max_in_leaf):
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
            child_node_index = (sorted_segm_keys[i] - node.sfc_start_key)/(node.number_sfc_keys/num_children)

            if child_node_index < 0 or child_node_index > num_children:
                raise RuntimeError("hilbert key out of bounds")

            child = (node + node.children_start + child_node_index)

            if child.number_segments == 0:
                child.particle_index_start = i

            # update the number of particles for child
            child.number_particles += self.num_part_leaf[i]
            child.number_segments += 1

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
                self._fill_segments_nodes(child, sorted_segm_keys, max_in_leaf)

    cdef void construct_global_tree(self, ParticleContainer pc, object comm):
        """
        Create a tree by recursively subdividing hilbert cuts.

        Parameters
        ----------
        pc : ParticleContainer
            Carray container holding all particle information
        comm : object
            Parallel class responsible for all communications
        """
        cdef np.ndarray arg, offsets
        cdef np.ndarray sendbuf, recvbuf
        cdef np.ndarray keys, sorted_keys
        cdef np.ndarray leaf_keys, leaf_part
        cdef np.ndarray glb_leaf_keys, glb_leaf_part

        cdef int i, j, size
        cdef int max_in_leaf
        cdef int glb_num_part, glb_num_leaves

        size = comm.Get_size()

        # collect number of particles from all process
        sendbuf = np.array([pc.get_number_of_particles()], dtype=np.int32)
        recvbuf = np.empty(size, dtype=np.int32)

        comm.Allgather(sendbuf=sendbuf, recvbuf=recvbuf)
        glb_num_part = np.sum(recvbuf)

        # sort hilbert keys
        keys = pc["keys"]
        sorted_keys = np.sort(keys)

        # create local tree first
        max_in_leaf = <int> (self.factor*glb_num_part/self.total_num_proc**2)
        self._build_local_tree(sorted_keys, max_in_leaf)

        # count local number of leaves
        self.num_leaves = self.mem_pool.number_leaves()

        # collect leaves from all process  
        leaf_keys = np.empty(self.num_leaves, dtype=np.int64)
        leaf_part = np.empty(self.num_leaves, dtype=np.int32)

        j = 0
        for i in range(self.mem_pool.used):
            self.mem_pool[i].children_start == -1:
                leaf_keys[j] = self.mem_pool[i].sfc_start_key
                leaf_part[j] = self.mem_pool[i].number_particles
                j += 1

        # prepare to bring all leafs form all local trees
        sendbuf[0] = self.num_leaves
        comm.Allgather(sendbuf=sendbuf, recvbuf=recvbuf)

        glb_num_leaves = np.sum(recvbuf)
        offsets = np.zeros(self.size)
        offsets[1:] = np.cumsum(recvbuf)[:-1]

        glb_leaf_keys = np.empty(glb_num_leaves, dtype=np.int64)
        glb_leaf_part = np.empty(glb_num_leaves, dtype=np.int32)

        comm.Allgatherv(leaf_keys, [glb_leaf_keys, recvbuf, offsets, MPI.INT64_T])
        comm.Allgatherv(leaf_part, [glb_leaf_part, recvbuf, offsets, MPI.INT])

        # sort global leaves
        ind = glb_leaf_keys.argsort()
        glb_leaf_keys[:] = glb_leaf_keys[ind]
        glb_leaf_part[:] = glb_leaf_part[ind]

        # rebuild tree using global leaves
        max_in_leaf = <int> (self.factor*glb_num_part/self.total_num_proc)
        self._build_global_tree(glb_num_part, glb_leaf_keys, max_in_leaf)

        # assign global leaves to array order
        self.num_leaves = 0
        self._leaves_to_array(self.root, &self.num_leaves)

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
                self._leaves_to_array(node + node.children_start + i)

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

    cdef int get_nearest_process_neighbors(self, double center[3], double h, np.int32_t[:] leaf_proc, int rank, LongArray nbrs):
        cdef int i
        cdef double smin[3]
        cdef double smax[3]

        # scale to hilbert coordinates
        for i in range(self.dim):
            smin[i] = ((center[i] - h) - self.domain_corner[i])*self.domain_fac
            smax[i] = ((center[i] + h) - self.domain_corner[i])*self.domain_fac

        self._neighbors(self.root, smin, smax, leaf_proc, rank, nbrs)

        # return number of processors found
        return nbrs.length

    cdef void neighbors(self, Node* node, double smin[3], double smax[3], np.int32_t[:] leaf_proc, int rank, LongArray nbrs):
        cdef int i

        # is node a leaf
        if node.children_start == -1:
            # dont count nodes on our process
            if rank != leaf_proc[node.array_index]:
                nbrs.append(leaf_proc[node.array_index])

        else:
            # does search box overlap with node
            for i in range(self.dim):
                if (node.center[i] + 0.5*node.box_length) < smin[i]: return
                if (node.center[i] - 0.5*node.box_length) > smax[i]: return

            # node overlaps open sub nodes
            for i in range(1 << self.dim):
                self._neighbors(node + node.children_start + i, smin, smax, leaf_proc, rank, nbrs)

    # temporary function to do outputs in python
    def dump_data(self):
        cdef list data_list = []
        cdef Node node

        cdef int i
        for i in range(self.mem_pool.used):
            node = self.mem_pool.node_array[i]
            if node.children_start == -1:
                data_list.append([node.center[0],
                    node.center[1],
                    node.center[2],
                    node.box_length,
                    node.number_particles])

        return data_list

cdef class QuadTree(BaseTree):

    def __init__(self, int total_num_part,
            np.ndarray[np.int64_t, ndim=1] sorted_part_keys,
            np.ndarray[np.float64_t, ndim=1] corner,
            double domain_length,
            np.ndarray[np.int64_t, ndim=1] sorted_segm_keys=None,
            np.ndarray[np.int32_t, ndim=1] num_part_leaf=None,
            int total_num_proc=1, double factor=1.0, int min_in_leaf=32, int order=21):

        BaseTree.__init__(self,
                total_num_part,
                sorted_part_keys,
                corner,
                domain_length,
                sorted_segm_keys,
                num_part_leaf,
                total_num_proc,
                factor,
                min_in_leaf,
                order)

        self.dim = 2
        self.hilbert_func = hilbert_key_2d

        # size of the domain is the size of hilbert space
        cdef int i
        for i in range(2):
            self.bounds[0][i] = 0
            self.bounds[1][i] = 2**(order)

        self.mem_pool = TreeMemoryPool(10000)

#cdef class OcTree(BaseTree):
#
#    def __init__(self, int total_num_part,
#            np.ndarray[np.int64_t, ndim=1] sorted_part_keys,
#            np.ndarray[np.float64_t, ndim=1] corner,
#            double domain_length,
#            np.ndarray[np.int64_t, ndim=1] sorted_segm_keys=None,
#            np.ndarray[np.int32_t, ndim=1] num_part_leaf=None,
#            int total_num_process=1, double factor=1.0, int order=21):
#
#        BaseTree.__init__(self,
#                total_num_part,
#                sorted_part_keys,
#                corner,
#                domain_length,
#                sorted_segm_keys,
#                num_part_leaf,
#                total_num_process,
#                factor,
#                order)
#
#        self.dim = 3
#        self.hilbert_func = hilbert_key_3d
#
#        # size of the domain is the size of hilbert space
#        cdef int i
#        for i in range(3):
#            self.bounds[0][i] = 0
#            self.bounds[1][i] = 2**(order)
#
#        self.mem_pool = TreeMemoryPool(100)
#
#    def update_particle_process(self, CarrayContainer pc, int rank, np.int32_t[:] leaf_procs):
#
#        cdef IntArray tags  = pc.get_carray("tag")
#        cdef LongArray proc = pc.get_carray("process")
#
#        cdef DoubleArray arr
#        cdef np.float64_t *x[3]
#        cdef np.int32_t pos[3]
#        cdef str field, axis
#        cdef np.int64_t key
#        cdef Node *node
#        cdef int i, j, boundary
#
#        for i, axis in enumerate("xyz"):
#            field = "position-" + axis
#            if field in pc.properties.keys():
#                arr = <DoubleArray> pc.get_carray(field)
#                x[i] = arr.get_data_ptr()
#
#        for i in range(pc.get_number_of_items()):
#
#            # map particle position into hilbert space
#            for j in range(self.dim):
#                pos[j] = <np.int32_t> ((x[j][i] - self.domain_corner[j])*self.domain_fac)
#                boundary += (pos[j] <= self.bounds[0][j] or self.bounds[1][j] <= pos[j])
#
#            # particle in the domain
#            if boundary == 0:
#                # generate hilbert key for particle
#                key = self.hilbert_func(pos[0], pos[1], pos[2], self.order)
#
#                # use key to find which leaf the particles lives in and store process id
#                node = self._find_leaf(key)
#                proc.data[i] = leaf_procs[node.array_index]
#
#            # particle outside domain
#            else:
#                proc.data[i] = -1
#
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
#        cdef int num_children = 1 << self.dim
#
#        while candidate.level < level and candidate.children_start != -1:
#            child_node_index = (key - candidate.sfc_start_key)/(candidate.number_sfc_keys/num_children)
#            candidate = candidate + candidate.children_start + child_node_index
#        return candidate
#
#    def collect_leaves_for_export(self):
#        """
#        For each leaf store the first key in the hilbert cut and the number of particles.
#
#        Returns
#        -------
#        start_keys : ndarray
#            Array that holds the start key for each leaf
#        num_part_leaf : ndarray
#            Array that holds the number of particles for each leaf
#        """
#        cdef int counter = 0
#
#        self.number_leaves = 0
#        self._count_leaves(self.root)
#
#        cdef np.int64_t[:]  start_keys    = np.empty(self.number_leaves, dtype=np.int64)
#        cdef np.int32_t[:]  num_part_leaf = np.empty(self.number_leaves, dtype=np.int32)
#
#        self._collect_leaves_for_export(self.root, &start_keys[0], &num_part_leaf[0], &counter)
#
#        return np.asarray(start_keys), np.asarray(num_part_leaf)
#
#    cdef void _collect_leaves_for_export(self, Node* node, np.int64_t *start_keys,
#            np.int32_t *num_part_leaf, int* counter):
#        """
#        Recursively store the first key in the hilbert cut and the number of particles
#        for each leaf.
#
#        Parameters
#        ----------
#        node : Node
#        start_keys : pointer to int64 array
#            Array that holds the start key for each leaf
#        num_part_leaf : pointer to int32 array
#            Array that holds the number of particles for each leaf
#        counter : pointer to int
#            Array index for start_keys and num_part_leaf
#        """
#        cdef int i
#        if node.children_start == -1:
#
#            start_keys[counter[0]] = node.sfc_start_key
#            num_part_leaf[counter[0]] = node.number_particles
#            counter[0] += 1
#
#        else:
#            for i in range(1 << self.dim):
#                self._collect_leaves_for_export(node + node.children_start + i,
#                        start_keys, num_part_leaf, counter)
#
#    def flag_migrate_particles(self, ParticleContainer pc, int my_proc, np.int32_t[:] leaf_procs):
#
#        cdef IntArray tags = pc.get_carray("tag")
#        cdef IntArray type = pc.get_carray("type")
#
#        cdef LongLongArray keys = pc.get_carray("key")
#        cdef LongArray proc = pc.get_carray("process")
#
#        cdef DoubleArray arr
#        cdef np.float64_t* x[3]
#        cdef np.int32_t pos[3]
#        cdef str axis, field
#        cdef np.int64_t key
#        cdef Node *node
#        cdef int i, j, boundary
#
#        for i, axis in enumerate("xyz"):
#            field = "position-" + axis
#            if field in pc.properties.keys():
#                arr = <DoubleArray> pc.get_carray(field)
#                x[i] = arr.get_data_ptr()
#
#        for i in range(pc.get_number_of_particles()):
#
#            # map particle position into hilbert space
#            for j in range(self.dim):
#                pos[j] = <np.int32_t> ((x[j][i] - self.domain_corner[j])*self.domain_fac)
#                boundary += (pos[j] <= self.bounds[0][j] or self.bounds[1][j] <= pos[j])
#
#            # make sure the key is in the global domain
#            if boundary == 0:
#                # generate hilbert key for particle
#                key = self.hilbert_func(pos[0], pos[1], pos[2], self.order)
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
#                # ghost particles outside the domain, store it
#                tags.data[i] = OldGhost
#                type.data[i] = Exterior
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
#    def __init__(self,
#            #int total_num_part,
#            #np.ndarray[np.int64_t, ndim=1] sorted_part_keys,
#            #np.ndarray[np.int64_t, ndim=1] sorted_segm_keys=None,
#            #np.ndarray[np.int32_t, ndim=1] num_part_leaf=None,
#            np.ndarray[np.float64_t, ndim=1] corner,
#            double domain_length,
#            int total_num_process=1, double factor=1.0, int min_in_leaf=32, int order=21):
#
#        self.order = order
#        self.min_in_leaf = min_in_leaf
#        self.factor = factor
#        self.root = NULL
#        self.hilbert_func = NULL
#
#        # original info of the particle domain
#        self.domain_corner = corner
#        self.domain_length = domain_length
#        self.domain_fac = (1 << order)/domain_length
#
#        #self.sorted_part_keys = sorted_part_keys
#        #self.total_num_part = total_num_part
#        #self.total_num_process = total_num_process
#
#
#        # building tree from particles
#        #if sorted_segm_keys is None:
#        #    self.build_using_cuts = 0
#        #    self.max_in_leaf = <int> (factor*total_num_part/total_num_process**2)
#
#        # building tree from hilbert segments
#        #else:
#        #    self.build_using_cuts = 1
#        #    self.num_part_leaf = num_part_leaf
#        #    self.sorted_segm_keys = sorted_segm_keys
#        #    self.max_in_leaf = <int> (factor*total_num_part/total_num_process)
#
#    def build_tree(self, max_in_leaf=None):
#        """
#        Create a tree by recursively subdividing hilbert cuts.
#
#        Parameters
#        ----------
#        max_in_leaf : int
#            max number of particles in a node, default is factor * total number
#            of particles / number of process
#        """
#        cdef int max_leaf
#        if max_in_leaf != None:
#            max_leaf = max_in_leaf
#        else:
#            max_leaf = self.max_in_leaf
#
#        self.root = self.mem_pool.get(1)
#
#        # create root node which holds all possible hilbert
#        # keys in a grid of 2^order resolution per dimension
#        self.root.children_start = -1
#        self.root.sfc_start_key = 0
#        self.root.number_sfc_keys = 2**(self.dim*self.order)
#        self.root.particle_index_start = 0
#        self.root.level = 0
#        self.root.box_length = 2**(self.order)
#
#        if self.build_using_cuts:
#            self.root.number_particles = self.total_num_part
#            self.root.number_segments  = self.sorted_segm_keys.shape[0]
#            self._fill_segments_nodes(self.root, max_leaf)
#        else:
#            self.root.number_particles = self.sorted_part_keys.shape[0]
#            self.root.number_segments = 0
#            self._fill_particles_nodes(self.root, max_leaf)
#
# >>>> delete this <<<<
#    cdef void _count_leaves(self, Node* node):
#        """
#        Recursively count the number of leaves by walking the tree.
#
#        Parameters
#        ----------
#        node : Node
#        """
#        cdef int i
#        if node.children_start == -1:
#            self.number_leaves += 1
#        else:
#            for i in range(1 << self.dim):
#                self._count_leaves(node + node.children_start + i)
#
#    def count_leaves(self):
#        """
#        Count the leaves in the tree.
#
#        Returns
#        -------
#        number_leaves : int
#            Number of leaves in the tree
#        """
#        self.number_leaves = 0
#        self._count_leaves(self.root)
#        return self.number_leaves
# >>>> delete this <<<<
#
#    cdef int _assign_leaves_to_array(self):
#        """
#        Map each leaf to an array index.
#
#        Returns
#        -------
#        number_leaves : int
#            Number of leaves in the tree.
#        """
#        #self.number_leaves = 0
#        self.num_leaves = 0
#        self._leaves_to_array(self.root)
#
#        #return self.number_leaves
#        return self.num_leaves
#
#    def count_nodes(self):
#        """
#        Count the number of nodes that are in the tree.
#
#        Returns
#        -------
#        number_nodes : int
#            Number of nodes in the tree
#        """
#        return self.mem_pool.used
