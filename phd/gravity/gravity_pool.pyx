cimport libc.stdlib as stdlib

cdef class GravityPool:

    def __init__(self, int num_nodes):
        self.array = <Node*> stdlib.malloc(num_nodes*sizeof(Node))
        if self.array == NULL:
            raise MemoryError()
        self.used = 0
        self.capacity = num_nodes

    cdef Node* get(self, int count):
        """
        Allocate count number of nodes from the pool and return
        pointer to the first node.

        Parameters
        ----------
        int : count
            Number of nodes to allocate

        Returns
        -------
        Node*
            Pointer to node allocated
        """
        cdef Node* first_node
        cdef int current = self.used

        if (self.used + count) > self.capacity:
            self.resize(2*self.capacity)
        first_node = &self.array[current]
        self.used += count

        return first_node

    cdef void resize(self, int size):
        """
        Resize the memory pool to have size number of nodes available
        for use. Note this does not mean there are size nodes used.

        Parameters
        ----------
        int : size
            Number of nodes allocated
        """
        cdef void* array = NULL
        if size > self.capacity:
            array = <Node*>stdlib.realloc(self.array, size*sizeof(Node))

            if array ==  NULL:
                stdlib.free(<void*>self.array)
                raise MemoryError('Insufficient Memory in gravity pool')

            self.array = <Node*> array
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
            if(self.array[i].flags & LEAF):
                num_leaves += 1
        return num_leaves

    cpdef int number_nodes(self):
        """
        Return number of nodes used from the pool.
        """
        return self.used

    def __dealloc__(self):
        stdlib.free(<void*>self.array)
