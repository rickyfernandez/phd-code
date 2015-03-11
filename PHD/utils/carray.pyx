import numpy as np

cimport numpy as np
cimport libc.stdlib as stdlib
cimport libc.string as string

# importing some Numpy C-api functions
cdef extern from "numpy/arrayobject.h":
    cdef void import_array()

    ctypedef struct PyArrayObject:
        char *data
        np.npy_intp *dimensions

    cdef enum NPY_TYPES:
        NPY_INT,
        NPY_UINT,
        NPY_LONG,
        NPY_FLOAT,
        NPY_DOUBLE

    np.ndarray PyArray_SimpleNewFromData(int, np.npy_intp*, int, void*)

# numpy module initialization call
import_array()


cdef class BaseArray:
    """Base class for managed C-arrays"""

    cdef readonly int length, alloc
    cdef np.ndarray _npy_array

    cpdef str get_c_type(self):
        """Return the c data type of this array."""
        raise NotImplementedError, 'BaseArray::get_c_type'

    cpdef reserve(self, int size):
        """Resizes the internal data to required size"""
        raise NotImplementedError, 'BaseArray::reserve'

    cpdef resize(self, int size):
        """Resizes the internal data to required size"""
        raise NotImplementedError, 'BaseArray::resize'

    cpdef np.ndarray get_npy_array(self):
        """returns a numpy array of the data: do not keep its reference"""
        return self._npy_array

    cpdef squeeze(self):
        """Release any unused memory."""
        raise NotImplementedError, 'BaseArray::squeeze'

    cpdef remove(self, np.ndarray index_list, int sorted_flag=0):
        """Remove the particles with indices in index_list."""
        raise NotImplementedError, 'BaseArray::remove'

    cpdef extend(self, np.ndarray in_array):
        """Extend the array with data from in_array."""
        raise NotImplementedError, 'BaseArray::extend'

    cpdef align_array(self, np.ndarray new_indices):
        """Rearrange the array contents according to the new indices."""
        raise NotImplementedError, 'BaseArray::align_array'

    cpdef reset(self):
        """Reset the length of the array to 0."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        self.length = 0
        arr.dimensions[0] = self.length

    def __len__(self):
        return self.length

    def __iter__(self):
        """Support the iteration protocol"""
        return BaseArrayIter(self)


cdef class BaseArrayIter:
    """Iteration object to support iteration over BaseArray."""
    def __init__(self, BaseArray arr):
        self.arr = arr
        self.i = -1

    def __next__(self):
        self.i = self.i+1
        if self.i < self.arr.length:
            return self.arr[self.i]
        else:
            raise StopIteration

    def __iter__(self):
        return self


cdef class DoubleArray(BaseArray):
    """Represents an array of doubles"""

    cdef double *data

    def __cinit__(self, int n=0):
        """Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(double)) and sets up
        the numpy array.

        Parameters:
        -----------
        n -- Length of the array.

        Data attributes:
        ----------------
        data -- Pointer to a double array.
        alloc -- Size of the data buffer allocated
        """
        self.length = n
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <double*> stdlib.malloc(n*sizeof(double))
        if self.data == <double*> NULL:
            raise MemoryError

        self._setup_npy_array()

    def __dealloc_(self):
        """Frees the c array"""
        stdlib.free(<void*>self.data)

    def __getitem__(self, int pid):
        """Get particle item at position pid."""
        return self.data[pid]

    def __setitem__(self, int pid, double value):
        """Set location pid to value."""
        self.data[pid] = value

    cdef _setup_npy_array(self):
        """Create numpy array of the data"""
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims,
                NPY_DOUBLE, self.data)

    cpdef str get_c_type(self):
        """Return the c data type for this array."""
        return 'double'

    cdef double* get_data_ptr(self):
        """Return the internal data pointer."""
        return self.data

    cpdef double get(self, int pid):
        """Get item at position pid."""
        return self.data[pid]

    cpdef set(self, int pid, double value):
        """Set location pid to value."""
        self.data[pid] = value

    cpdef append(self, double value):
        """Appends value to the end of the array."""
        cdef int l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, int size):
        """Resizes the internal data to size*sizeof(double) bytes."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <double*> stdlib.realloc(self.data, size*sizeof(double))

            if data == NULL:
                stdlib.free(<void*> self.data)
                raise MemoryError

            self.data = <double*> data
            self.alloc = size
            arr.data = <char*> self.data

    cpdef resize(self, int size):
        """Resizes internal data to size*sizeof(double) bytes."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """Release any unused memory."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        data = <double*> stdlib.realloc(self.data, self.length*sizeof(double))

        if data == NULL:
            stdlib.free(<void*>self.data)
            raise MemoryError

        self.data = <double*> data
        self.alloc = self.length
        arr.data = <char*> self.data

    cpdef remove(self, np.ndarray index_list, int sorted_flag=0):
        """Remove the particles with indices in index_list.

        Parameters
        ----------
        index_list -- a list of indices which should be removed.
        sorted_flag -- indicates if the input is sorted in ascending order."""
        cdef int i
        cdef int inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef int pid
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if inlength > self.length:
            return

        if sorted_flag != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        for i in xrange(inlength):
            pid = sorted_indices[inlength-(i+1)]
            if pid < self.length:
                self.data[pid] = self.data[self.length-1]
                self.length -= 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """Extend the array with data from in_array.

        Parameters
        ----------
        in_array -- a numpy array with data to be added to the current array."""
        cdef int length = in_array.size
        cdef int i
        for i in xrange(length):
            self.append(in_array[i])

    cpdef align_array(self, np.ndarray new_indices):
        """Rearrange the array contents according to the new indices."""
        if new_indices.size != self.length:
            raise ValueError, 'Unequal array lengths'

        cdef int i
        cdef int length = self.length
        cdef long n_bytes
        cdef double *temp

        n_bytes = sizeof(double)*length
        temp = <double*> stdlib.malloc(n_bytes)

        string.memcpy(<void*> temp, <void*> self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i in xrange(length):
            if i != new_indices[i]:
                self.data[i] = temp[new_indices[i]]

        stdlib.free(<void*> temp)


cdef class IntArray(BaseArray):
    """Represents an array of ints"""

    cdef int *data

    def __cinit__(self, int n=0):
        """Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(int)) and sets up
        the numpy array.

        Parameters:
        -----------
        n -- Length of the array.

        Data attributes:
        ----------------
        data -- Pointer to an integer array.
        alloc -- Size of the data buffer allocated
        """
        self.length = n
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <int*> stdlib.malloc(n*sizeof(int))
        if self.data == <int*> NULL:
            raise MemoryError

        self._setup_npy_array()

    def __dealloc_(self):
        """Frees the c array"""
        stdlib.free(<void*>self.data)

    def __getitem__(self, int pid):
        """Get item at position pid."""
        return self.data[pid]

    def __setitem__(self, int pid, int value):
        """Set location pid to value."""
        self.data[pid] = value

    cdef _setup_npy_array(self):
        """Create numpy array of the data"""
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims,
                NPY_INT, self.data)
#
#    cpdef str get_c_type(self):
#        """Return the c data type for this array."""
#        return 'int'
#
#    cdef int* get_data_ptr(self):
#        """Return the internal data pointer."""
#        return self.data
#
    cpdef int get(self, int pid):
        """Get item at position pid."""
        return self.data[pid]

    cpdef set(self, int pid, int value):
        """Set location pid to value."""
        self.data[pid] = value

    cpdef append(self, int value):
        """Appends value to the end of the array."""
        cdef int l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, int size):
        """Resizes the internal data to size*sizeof(int) bytes."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <double*> stdlib.realloc(self.data, size*sizeof(int))

            if data == NULL:
                stdlib.free(<void*> self.data)
                raise MemoryError

            self.data = <int*> data
            self.alloc = size
            arr.data = <char*> self.data

    cpdef resize(self, int size):
        """Resizes internal data to size*sizeof(double) bytes."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """Release any unused memory."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        data = <int*> stdlib.realloc(self.data, self.length*sizeof(int))

        if data == NULL:
            stdlib.free(<void*>self.data)
            raise MemoryError

        self.data = <int*> data
        self.alloc = self.length
        arr.data = <char*> self.data

    cpdef remove(self, np.ndarray index_list, int sorted_flag=0):
        """Remove the particles with indices in index_list.

        Parameters
        ----------
        index_list -- a list of indices which should be removed.
        sorted_flag -- indicates if the input is sorted in ascending order."""
        cdef int i
        cdef int inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef int pid
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if inlength > self.length:
            return

        if sorted_flag != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        for i in xrange(inlength):
            pid = sorted_indices[inlength-(i+1)]
            if pid < self.length:
                self.data[pid] = self.data[self.length-1]
                self.length = self.length - 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """Extend the array with data from in_array.

        Parameters
        ----------
        in_array -- a numpy array with data to be added to the current array."""
        cdef int length = in_array.size
        cdef int i
        for i in xrange(length):
            self.append(in_array[i])

    cpdef align_array(self, np.ndarray new_indices):
        """Rearrange the array contents according to the new indices."""
        if new_indices.size != self.length:
            raise ValueError, 'Unequal array lengths'

        cdef int i
        cdef int length = self.length
        cdef long n_bytes
        cdef int *temp

        n_bytes = sizeof(int)*length
        temp = <int*> stdlib.malloc(n_bytes)

        string.memcpy(<void*> temp, <void*> self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i in xrange(length):
            if i != new_indices[i]:
                self.data[i] = temp[new_indices[i]]

        stdlib.free(<void*> temp)
