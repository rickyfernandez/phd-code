import numpy as np

cimport numpy as np
cimport libc.stdlib as stdlib
cimport cython

key_index_2d = [0, 1, 3, 2]

cpdef np.uint64_t hilbert_key_2d(np.uint32_t x, np.uint32_t y, int order):

    cdef np.uint64_t key = 0
    cdef int i, xbit, ybit
    for i in range(order-1, -1, -1):

        xbit = 1 if x & (1 << i) else 0
        ybit = 1 if y & (1 << i) else 0

        if   xbit == 0 and ybit == 0: x, y = y, x
        elif xbit == 1 and ybit == 0: x, y = ~y, ~x

        key = (key << 2) + key_index_2d[(xbit << 1) + ybit]

    return key


cdef struct Oct:

    np.uint64_t sfc_key     # space filling curve key for oct
    np.uint32_t level       # level of octree

    np.uint64_t sfc_start_key         # first key in space filling curve cut
    np.uint64_t number_sfc_keys       # total number of possible space filling keys in this oct

    np.int64_t particle_index_start   # index of first particle in space filling curve cut
    np.uint64_t number_particles      # number of particles in cut

    int leaf               # is this oct a leaf

    np.float64_t box_length  # side length of oct
    np.float64_t center[2]   # center of coordinates of oct

    Oct*  parent           # parent of oct
    Oct* children          # children nodes, 4 of them


cdef class Octree:

    cdef readonly np.ndarray sorted_particle_keys
    cdef readonly int max_leaf_particles
    cdef readonly int order

    cdef Oct* root

    def __init__(self, sorted_particle_keys, max_leaf_particles, order):

        self.sorted_particle_keys = np.ascontiguousarray(sorted_particle_keys, dtype=np.int64)
        self.order = order

        self.max_leaf_particles = max_leaf_particles

        # build root
        self.root = <Oct*>stdlib.malloc(sizeof(Oct))
        if self.root == <Oct*> NULL:
            raise MemoryError

        self.root.children = NULL
        self.root.sfc_start_key = 0
        self.root.number_sfc_keys = 2**(2*order)
        self.root.particle_index_start = 0
        self.root.number_particles = sorted_particle_keys.shape[0]
        self.root.level = 0
        self.root.leaf = 1
        self.root.box_length = 2**(order)

        cdef int i
        for i in range(2):
            self.root.center[i] = 0.5*2**(order)

        # build octree
        self.build_tree()


    cdef void build_tree(self):
        self.fill_particles_in_oct(self.root)


    cdef void fill_particles_in_oct(self, Oct* o):

        # create oct children
        o.children = <Oct*>stdlib.malloc(sizeof(Oct)*4)
        if o.children == <Oct*> NULL:
            raise MemoryError

        # pass parent data to children 
        cdef int i, j
        for i in range(4):

            o.children[i].parent = o
            o.children[i].leaf = 1
            o.children[i].level = o.level + 1
            o.children[i].number_sfc_keys = o.number_sfc_keys/4
            o.children[i].sfc_start_key   = o.sfc_start_key + i*o.number_sfc_keys/4
            o.children[i].particle_index_start = -1
            o.children[i].number_particles = 0
            o.children[i].box_length = o.box_length/2.0
            o.children[i].children = NULL


        cdef  np.uint64_t key
        cdef int child_oct_index
        for i in range(2):
            for j in range(2):

                key = hilbert_key_2d( <np.uint32_t> (o.center[0] + (2*i-1)*o.box_length/4.0),
                        <np.uint32_t> (o.center[1] + (2*j-1)*o.box_length/4.0), self.order)

                # which oct does this key belong to
                child_oct_index = (key - o.sfc_start_key)/(o.number_sfc_keys/4)
                o.children[child_oct_index].sfc_key = key
                o.children[child_oct_index].center[0] = o.center[0] + (2*i-1)*o.box_length/4.0
                o.children[child_oct_index].center[1] = o.center[1] + (2*j-1)*o.box_length/4.0

        # loop over parent particles and assign them to proper child
        for i in range(o.particle_index_start, o.particle_index_start + o.number_particles):

            # which oct does this particle belong to
            child_oct_index = (self.sorted_particle_keys[i] - o.sfc_start_key)/(o.number_sfc_keys/4)

            # if child node is empty then this is the first particle in the cut
            if o.children[child_oct_index].number_particles == 0:
                o.children[child_oct_index].particle_index_start = i

            o.children[child_oct_index].number_particles += 1

        # parent is no longer a leaf
        o.leaf = 0

        # if child has more particles then the maximum allowed, then subdivide 
        for i in range(4):
            if o.children[i].number_particles > self.max_leaf_particles:
                self.fill_particles_in_oct(<Oct*> &o.children[i])


    def oct_neighbor_search(self):
        pass


    cdef iterate(self, Oct* o, list data_list):

        data_list.append([o.center[0], o.center[1], o.box_length,
            o.level, o.particle_index_start, o.number_particles])

        cdef int i
        if o.children != NULL:
            for i in range(4):
                self.iterate(&o.children[i], data_list)


    def dump_data(self):
        cdef list data_list = []
        self.iterate(self.root, data_list)
        return data_list


#    cdef free_octs(self, Oct* o):
#        cdef int i
#        for i in range(4):
#            if o.children[i].leaf == 0:
#                self.free_octs(&o.children[i])
#            stdlib.free(o.children)
#
#
#    def __dealloc__(self):
#        if self.root.children != NULL:
#            self.free_octs(self.root)
#        stdlib.free(self.root)
