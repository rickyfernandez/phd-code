cimport libc.stdlib as stdlib
from .gravity_tree cimport LEAF

cdef class GravityPool:
    """
    Memory pool of gravity nodes for tree construction.
    """
    def __init__(self, int num_nodes):
        """
        Initialization memory pool, which reserves num_nodes nodes
        in pool for use.

        Parameters
        ----------
        num_nodes : int
            Number of nodees to reserve for pool
        """
        self.array = <Node*> stdlib.malloc(num_nodes*sizeof(Node))
        if self.array == NULL:
            raise MemoryError("Insufficient memory on gravity pool inilization")
        self.used = 0
        self.capacity = num_nodes

    cdef Node* get(self, int count):
        """
        Allocate count number of nodes from pool and return
        pointer to first node of allocation.

        Parameters
        ----------
        count : int
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
        size : size
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
        Reset the pool, no nodes used
        """
        self.used = 0

    cpdef int number_leafs(self):
        """
        Return number of nodes flagged as leafs in pool

        Returns
        -------
        int
            Number of leafs in pool
        """
        cdef int i, num_leafs = 0
        for i in range(self.used):
            if(self.array[i].flags & LEAF):
                num_leafs += 1
        return num_leafs

    cpdef int number_nodes(self):
        """
        Return number of nodes used in pool.

        Returns
        -------
        int
            Number of nodes used in pool
        """
        return self.used

    def __dealloc__(self):
        """
        Deallocate array of nodes used in pool
        """
        stdlib.free(<void*>self.array)
