import numpy as np

cimport numpy as np
cimport libc.stdlib as stdlib
cimport cython

cdef struct Oct:

    np.uint64_t sfc_key     # space filling curve key for oct at given level 
    np.uint32_t level       # level of octree

    np.uint64_t sfc_start_key         # first key in space filling curve cut
    np.uint64_t number_sfc_keys       # total number of space filling keys in this oct

    np.uint64_t particle_index_start  # index of first particle in space filling curve cut
    np.uint64_t number_particles      # number of particles in cut

    int leaf               # is this oct a leaf

    np.int64_t box_length  # side length of oct
    np.int64_t center[3]   # center of coordinates of oct

    Oct*  parent           # parent of oct
    Oct** children         # children nodes, 8 of them


cdef class Octree:

    cdef readonly np.ndarray sorted_particle_keys

    cdef np.int32_t max_leaf_particles
    cdef Oct* root

    def __init__(self, sorted_particle_keys, domain_center, domain_size, total_number_of_sfc_keys):

        self.sorted_particle_keys = np.ascontiguousarray(sorted_particle_keys, dtype=np.int64)

        self.root = NULL
        self.max_leaf_particles = 4

        # build root
        self.root = <Oct*>stdlib.malloc(sizeof(Oct))
        if self.root == <Oct*> NULL:
            raise MemoryError

        self.root.sfc_key = 0
        self.root.number_sfc_keys = total_number_of_sfc_keys
        self.root.particle_index_start = 0
        self.root.number_particles = sorted_particle_keys.shape[0]
        self.root.leaf = 1
        self.root.box_length = domain_size

        cdef int i
        for i in range(3):
            self.root.center[i] = domain_center[i]

        # build octree
        self.build_tree()


    cdef void build_tree(self):
        self.fill_particles_in_oct(self.root)


    cdef void fill_particles_in_oct(self, Oct* o):

        # create oct children
        o.children = <Oct**> stdlib.malloc(sizeof(void*)*4)
        if o.children == <Oct**> NULL:
            raise MemoryError

        # pass parent data to children 
        cdef int i, j
        for i in range(4):

            o.children[i].parent = o
            o.children[i].leaf = 1
            o.children[i].level = o.level + 1
            o.children[i].number_sfc_keys = o.number_sfc_keys/4
            o.children[i].sfc_start_key   = o.sfc_start_key + i*o.number_sfc_keys/4
            o.children[i].number_particles = 0
            o.children[i].box_length = o.box_length/2.0

        cdef  np.uint64_t key
        cdef int child_oct_index
        for i in range(2):
            for j in range(2):

                key = hilbert(o.center[0] + (2*i-1)*o.box_length/2,
                        o.center[i] + (2*j-1)*o.box_length/2, self.order)

                # which oct does this key belong to
                child_oct_index = (key - o.sfc_start_key)/(o.number_sfc_keys/4)
                o.children[child_oct_index].sfc_key = key
                o.children[child_oct_index].center[0] = o.center[0] + (2*i-1)*o.box_length/2
                o.children[child_oct_index].center[1] = o.center[1] + (2*j-1)*o.box_length/2

        # loop over parent particles and assign them to proper child
        for i in range(o.particle_index_start, o.number_particles):

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
            if o.children[i].number_particles > self.leaf_max_particles:
                self.fill_particles_in_oct(o.children[i])


    def oct_neighbor_search(self):
        pass


##    cdef iterate(self, Oct* o, list data_list):
##
##        data_list.append([o.center, o.box_length])
##
##        cdef int i
##        for i in range(4):
##            if o.children[i] == NULL:
##                self.iterate(o.children[i], data_list)
##
##
##    def dump_data(self):
##        list data_list = []
##        self.iterate(self.root, data_list)
##
##
##    cdef free_octs(self, Oct* o):
##        cdef int i
##        for i in range(4):
##            if o.children[i].leaf == 0:
##                self.free_octs(o.children[i])
##            stdlib.free(o)
##
##
##    def __dealloc__(self):
##        if self.root == NULL:
##            return
##        self.free_octs(self.root)
