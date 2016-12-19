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

    np.ndarray PyArray_SimpleNewFromData(int, np.npy_intp*, int, void*)

# numpy module initialization call
import_array()


cdef class BaseArray:
    """Base class for managed C-arrays."""

    cpdef str get_c_type(self):
        """Return the c data type of this array."""
        raise NotImplementedError, 'BaseArray::get_c_type'

    cpdef reserve(self, long size):
        """Resizes the internal data to required size."""
        raise NotImplementedError, 'BaseArray::reserve'

    cpdef resize(self, long size):
        """Resizes the internal data to required size."""
        raise NotImplementedError, 'BaseArray::resize'

    cpdef np.ndarray get_npy_array(self):
        """Returns a numpy array of the data: do not keep its reference."""
        return self._npy_array

    cpdef squeeze(self):
        """Release any unused memory."""
        raise NotImplementedError, 'BaseArray::squeeze'

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0):
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

    cpdef shrink(self, long size):
        """Reset the length of the array to length size."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if size > self.length:
            raise ValueError, 'shrink size is larger then array size'
        self.length = size
        arr.dimensions[0] = self.length

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """"Copy values of indexed particles from self to dest."""

    def __len__(self):
        return self.length

    def __iter__(self):
        """Support the iteration protocol."""
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
    """Represents an array of 64 bit floats."""

    def __cinit__(self, long n=0):
        """
        Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(np.float64_t)) and sets up
        the numpy array.

        Parameters:
        -----------
        n : int
            Length of the initial buffer.

        Data attributes:
        ----------------
        data : np.float64_t*
            Pointer to np.float64 buffer.
        alloc : int
            Size of the data buffer allocated.
        length : int
            Number of slots used in the buffer.
        """
        self.length = n
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <np.float64_t*> stdlib.malloc(n*sizeof(np.float64_t))
        if self.data == <np.float64_t*> NULL:
            raise MemoryError

        self._setup_npy_array()

    def __dealloc__(self):
        """Frees the c array."""
        stdlib.free(<void*>self.data)

    def __getitem__(self, long pid):
        """Get particle item at position pid."""
        return self.data[pid]

    def __setitem__(self, long pid, np.float64_t value):
        """Set location pid to value."""
        self.data[pid] = value

    cdef _setup_npy_array(self):
        """Create numpy array of the data."""
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims,
                np.NPY_FLOAT64, self.data)

    cpdef str get_c_type(self):
        """Return the c data type for this array."""
        return 'np.float64'

    cdef np.float64_t* get_data_ptr(self):
        """Return the internal data pointer."""
        return self.data

    cpdef np.float64_t get(self, long pid):
        """Get item at position pid."""
        return self.data[pid]

    cpdef set(self, long pid, np.float64_t value):
        """Set location pid to value."""
        self.data[pid] = value

    cpdef append(self, np.float64_t value):
        """Appends value to the end of the array."""
        cdef long l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, long size):
        """Resizes the internal data to size*sizeof(np.float64_t) bytes."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <np.float64_t*> stdlib.realloc(self.data, size*sizeof(np.float64_t))

            if data == NULL:
                stdlib.free(<void*> self.data)
                raise MemoryError

            self.data = <np.float64_t*> data
            self.alloc = size
            arr.data = <char*> self.data

    cpdef resize(self, long size):
        """
        Resizes internal data to size*sizeof(np.float64_t) bytes
        and sets the length to the new size.
        """
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """Release any unused memory."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        data = <np.float64_t*> stdlib.realloc(self.data, self.length*sizeof(np.float64_t))

        if data == NULL:
            # free original data
            stdlib.free(<void*> self.data)
            raise MemoryError

        self.data = <np.float64_t*> data
        self.alloc = self.length
        arr.data = <char*> self.data

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0):
        """
        Remove the particles with indices in index_list.

        Parameters
        ----------
        index_list : np.ndarray
            Indices which should be removed.
        input_sorted : bint
            Indicates if the input is sorted in ascending order. If not
            the array will be sorted internally.
        """
        cdef int i
        cdef int inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef int pid
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        for i in range(inlength):
            pid = sorted_indices[inlength-(i+1)]
            if pid < self.length:
                self.data[pid] = self.data[self.length-1]
                self.length -= 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.

        Parameters
        ----------
        in_array : ndarray
            Array with data to be added to the current array.
        """
        cdef int length = in_array.size
        cdef int i
        for i in range(length):
            self.append(in_array[i])

    cpdef align_array(self, np.ndarray new_indices):
        """Rearrange the array contents according to the new indices."""
        if new_indices.size != self.length:
            raise ValueError, 'Unequal array lengths'

        cdef int i
        cdef int length = self.length
        cdef int n_bytes
        cdef np.float64_t *temp

        n_bytes = sizeof(np.float64_t)*length
        temp = <np.float64_t*> stdlib.malloc(n_bytes)

        string.memcpy(<void*> temp, <void*> self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i in range(length):
            if i != new_indices[i]:
                self.data[i] = temp[new_indices[i]]

        stdlib.free(<void*> temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
        Copies values of indices in indices from self to dest.

        no size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        cdef DoubleArray dest_array = <DoubleArray>dest
        cdef int i

        for i in range(indices.length):
            dest_array.data[i] = self.data[indices.data[i]]

cdef class IntArray(BaseArray):
    """Represents an array of 8 bit integers."""

    def __cinit__(self, long n=0):
        """
        Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(np.int8_t)) and sets up
        the numpy array.

        Parameters:
        -----------
        n : int
            Length of the initial buffer.

        Data attributes:
        ----------------
        data : np.int8_t*
            Pointer to np.int8t buffer.
        alloc : int
            Size of the data buffer allocated
        length : int
            Number of slots used in the buffer
        """
        self.length = n
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <np.int8_t*> stdlib.malloc(n*sizeof(np.int8_t))
        if self.data == <np.int8_t*> NULL:
            raise MemoryError

        self._setup_npy_array()

    def __dealloc__(self):
        """Frees the c array."""
        stdlib.free(<void*>self.data)

    def __getitem__(self, long pid):
        """Get item at position pid."""
        return self.data[pid]

    def __setitem__(self, long pid, np.int8_t value):
        """Set location pid to value."""
        self.data[pid] = value

    cdef _setup_npy_array(self):
        """Create numpy array of the data."""
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims,
                np.NPY_INT8, self.data)

    cpdef str get_c_type(self):
        """Return the c data type for this array."""
        return 'np.int8'

    cdef np.int8_t* get_data_ptr(self):
        """Return the internal data pointer."""
        return self.data

    cpdef np.int8_t get(self, long pid):
        """Get item at position pid."""
        return self.data[pid]

    cpdef set(self, long pid, np.int8_t value):
        """Set location pid to value."""
        self.data[pid] = value

    cpdef append(self, np.int8_t value):
        """Appends value to the end of the array."""
        cdef int l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, long size):
        """Resizes the internal data to size*sizeof(np.int8_t) bytes."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <np.int8_t*> stdlib.realloc(self.data, size*sizeof(np.int8_t))

            if data == NULL:
                stdlib.free(<void*> self.data)
                raise MemoryError

            self.data = <np.int8_t*> data
            self.alloc = size
            arr.data = <char*> self.data

    cpdef resize(self, long size):
        """
        Resizes internal data to size*sizeof(np.int8_t) bytes
        and sets the length to the new size.
        """
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """Release any unused memory."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        data = <np.int8_t*> stdlib.realloc(self.data, self.length*sizeof(np.int8_t))

        if data == NULL:
            # free original data
            stdlib.free(<void*>self.data)
            raise MemoryError

        self.data = <np.int8_t*> data
        self.alloc = self.length
        arr.data = <char*> self.data

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0):
        """
        Remove the particles with indices in index_list.

        Parameters
        ----------
        index_list : np.ndarray
            Indices which should be removed.
        input : bint
            Indicates if the input is sorted in ascending order. If not
            the array will be sorted internally.
        """
        cdef int i
        cdef int inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef int pid
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        for i in range(inlength):
            pid = sorted_indices[inlength-(i+1)]
            if pid < self.length:
                self.data[pid] = self.data[self.length-1]
                self.length = self.length - 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.

        Parameters
        ----------
        in_array : ndarray
            Array with data to be added to the current array.
        """
        cdef long length = in_array.size
        cdef long i
        for i in range(length):
            self.append(in_array[i])

    cpdef align_array(self, np.ndarray new_indices):
        """Rearrange the array contents according to the new indices."""
        if new_indices.size != self.length:
            raise ValueError, 'Unequal array lengths'

        cdef long i
        cdef long length = self.length
        cdef long n_bytes
        cdef np.int8_t *temp

        n_bytes = sizeof(np.int8_t)*length
        temp = <np.int8_t*> stdlib.malloc(n_bytes)

        string.memcpy(<void*> temp, <void*> self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i in range(length):
            if i != new_indices[i]:
                self.data[i] = temp[new_indices[i]]

        stdlib.free(<void*> temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
        Copies values of indices in indices from self to dest.

        no size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        cdef IntArray dest_array = <IntArray>dest
        cdef int i

        for i in range(indices.length):
            dest_array.data[i] = self.data[indices.data[i]]

cdef class LongArray(BaseArray):
    """Represents an array of np.int32_t."""

    def __cinit__(self, long n=0):
        """
        Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(np.int32_t)) and sets up
        the numpy array.

        Parameters:
        -----------
        n : int
            Length of the initial buffer.

        Data attributes:
        ----------------
        data : np.int32_t*
            Pointer to np.int32 buffer.
        alloc : int
            Size of the data buffer allocated
        length : int
            Number of slots used in the buffer
        """
        self.length = n
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <np.int32_t*> stdlib.malloc(n*sizeof(np.int32_t))
        if self.data == <np.int32_t*> NULL:
            raise MemoryError

        self._setup_npy_array()

    def __dealloc__(self):
        """Frees the c array."""
        stdlib.free(<void*>self.data)

    def __getitem__(self, long pid):
        """Get particle item at position pid."""
        return self.data[pid]

    def __setitem__(self, long pid, np.int32_t value):
        """Set location pid to value."""
        self.data[pid] = value

    cdef _setup_npy_array(self):
        """Create numpy array of the data."""
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims,
                np.NPY_INT32, self.data)

    cpdef str get_c_type(self):
        """Return the c data type for this array."""
        return 'np.int32_t'

    cdef np.int32_t* get_data_ptr(self):
        """Return the internal data pointer."""
        return self.data

    cpdef np.int32_t get(self, long pid):
        """Get item at position pid."""
        return self.data[pid]

    cpdef set(self, long pid, np.int32_t value):
        """Set location pid to value."""
        self.data[pid] = value

    cpdef append(self, np.int32_t value):
        """Appends value to the end of the array."""
        cdef long l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, long size):
        """Resizes the internal data to size*sizeof(np.float32_t) bytes."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <np.int32_t*> stdlib.realloc(self.data, size*sizeof(np.int32_t))

            if data == NULL:
                stdlib.free(<void*> self.data)
                raise MemoryError

            self.data = <np.int32_t*> data
            self.alloc = size
            arr.data = <char*> self.data

    cpdef resize(self, long size):
        """
        Resizes internal data to size*sizeof(np.int32_t) bytes
        and sets the length to the new size.
        """
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """Release any unused memory."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        data = <np.int32_t*> stdlib.realloc(self.data, self.length*sizeof(np.int32_t))

        if data == NULL:
            # free original data
            stdlib.free(<void*>self.data)
            raise MemoryError

        self.data = <np.int32_t*> data
        self.alloc = self.length
        arr.data = <char*> self.data

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0):
        """
        Remove the particles with indices in index_list.

        Parameters
        ----------
        index_list : np.ndarray
            Indices which should be removed.
        input_sorted : bint
            Indicates if the input is sorted in ascending order. If not
            the array will be sorted internally.
        """
        cdef long i
        cdef long inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long pid
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        for i in range(inlength):
            pid = sorted_indices[inlength-(i+1)]
            if pid < self.length:
                self.data[pid] = self.data[self.length-1]
                self.length -= 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.

        Parameters
        ----------
        in_array : ndarray
            Array with data to be added to the current array.
        """
        cdef long length = in_array.size
        cdef long i
        for i in range(length):
            self.append(in_array[i])

    cpdef align_array(self, np.ndarray new_indices):
        """Rearrange the array contents according to the new indices."""
        if new_indices.size != self.length:
            raise ValueError, 'Unequal array lengths'

        cdef long i
        cdef long length = self.length
        cdef long n_bytes
        cdef np.int32_t *temp

        n_bytes = sizeof(np.int32_t)*length
        temp = <np.int32_t*> stdlib.malloc(n_bytes)

        string.memcpy(<void*> temp, <void*> self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i in range(length):
            if i != new_indices[i]:
                self.data[i] = temp[new_indices[i]]

        stdlib.free(<void*> temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
        Copies values of indices in indices from self to dest.

        no size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        cdef LongArray dest_array = <LongArray>dest
        cdef long i

        for i in range(indices.length):
            dest_array.data[i] = self.data[indices.data[i]]

cdef class LongLongArray(BaseArray):
    """Represents an array of np.int64_t."""

    def __cinit__(self, long n=0):
        """
        Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(np.int64_t)) and sets up
        the numpy array.

        Parameters:
        -----------
        n : int
            Length of the initial buffer.

        Data attributes:
        ----------------
        data : np.int64_t*
            Pointer to np.int64 buffer.
        alloc : int
            Size of the data buffer allocated
        length : int
            Number of slots used in the buffer
        """
        self.length = n
        if n == 0:
            n = 16
        self.alloc = n
        self.data = <np.int64_t*> stdlib.malloc(n*sizeof(np.int64_t))
        if self.data == <np.int64_t*> NULL:
            raise MemoryError

        self._setup_npy_array()

    def __dealloc__(self):
        """Frees the c array."""
        stdlib.free(<void*>self.data)

    def __getitem__(self, long pid):
        """Get particle item at position pid."""
        return self.data[pid]

    def __setitem__(self, long pid, np.int64_t value):
        """Set location pid to value."""
        self.data[pid] = value

    cdef _setup_npy_array(self):
        """Create numpy array of the data."""
        cdef int nd = 1
        cdef np.npy_intp dims = self.length

        self._npy_array = PyArray_SimpleNewFromData(nd, &dims,
                np.NPY_INT64, self.data)

    cpdef str get_c_type(self):
        """Return the c data type for this array."""
        return 'np.int64_t'

    cdef np.int64_t* get_data_ptr(self):
        """Return the internal data pointer."""
        return self.data

    cpdef np.int64_t get(self, long pid):
        """Get item at position pid."""
        return self.data[pid]

    cpdef set(self, long pid, np.int64_t value):
        """Set location pid to value."""
        self.data[pid] = value

    cpdef append(self, np.int64_t value):
        """Appends value to the end of the array."""
        cdef long l = self.length
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if l >= self.alloc:
            self.reserve(l*2)
        self.data[l] = value
        self.length += 1

        # update the numpy arrays length
        arr.dimensions[0] = self.length

    cpdef reserve(self, long size):
        """Resizes the internal data to size*sizeof(np.float64_t) bytes."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        if size > self.alloc:
            data = <np.int64_t*> stdlib.realloc(self.data, size*sizeof(np.int64_t))

            if data == NULL:
                stdlib.free(<void*> self.data)
                raise MemoryError

            self.data = <np.int64_t*> data
            self.alloc = size
            arr.data = <char*> self.data

    cpdef resize(self, long size):
        """
        Resizes internal data to size*sizeof(np.int64_t) bytes
        and sets the length to the new size.
        """
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        # reserve memory
        self.reserve(size)

        # update the lengths
        self.length = size
        arr.dimensions[0] = self.length

    cpdef squeeze(self):
        """Release any unused memory."""
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array
        cdef void* data = NULL
        data = <np.int64_t*> stdlib.realloc(self.data, self.length*sizeof(np.int64_t))

        if data == NULL:
            # free original data
            stdlib.free(<void*>self.data)
            raise MemoryError

        self.data = <np.int64_t*> data
        self.alloc = self.length
        arr.data = <char*> self.data

    cpdef remove(self, np.ndarray index_list, bint input_sorted=0):
        """
        Remove the particles with indices in index_list.

        Parameters
        ----------
        index_list : np.ndarray
            Indices which should be removed.
        input_sorted : bint
            Indicates if the input is sorted in ascending order. If not
            the array will be sorted internally.
        """
        cdef long i
        cdef long inlength = index_list.size
        cdef np.ndarray sorted_indices
        cdef long pid
        cdef PyArrayObject* arr = <PyArrayObject*> self._npy_array

        if inlength > self.length:
            return

        if input_sorted != 1:
            sorted_indices = np.sort(index_list)
        else:
            sorted_indices = index_list

        for i in range(inlength):
            pid = sorted_indices[inlength-(i+1)]
            if pid < self.length:
                self.data[pid] = self.data[self.length-1]
                self.length -= 1
                arr.dimensions[0] = self.length

    cpdef extend(self, np.ndarray in_array):
        """
        Extend the array with data from in_array.

        Parameters
        ----------
        in_array : ndarray
            Array with data to be added to the current array.
        """
        cdef long length = in_array.size
        cdef long i
        for i in range(length):
            self.append(in_array[i])

    cpdef align_array(self, np.ndarray new_indices):
        """Rearrange the array contents according to the new indices."""
        if new_indices.size != self.length:
            raise ValueError, 'Unequal array lengths'

        cdef long i
        cdef long length = self.length
        cdef long n_bytes
        cdef np.int64_t *temp

        n_bytes = sizeof(np.int64_t)*length
        temp = <np.int64_t*> stdlib.malloc(n_bytes)

        string.memcpy(<void*> temp, <void*> self.data, n_bytes)

        # copy the data from the resized portion to the actual positions.
        for i in range(length):
            if i != new_indices[i]:
                self.data[i] = temp[new_indices[i]]

        stdlib.free(<void*> temp)

    cpdef copy_values(self, LongArray indices, BaseArray dest):
        """
        Copies values of indices in indices from self to dest.

        no size check if performed, we assume the dest to of proper size
        i.e. atleast as long as indices.
        """
        cdef LongLongArray dest_array = <LongLongArray>dest
        cdef long i

        for i in range(indices.length):
            dest_array.data[i] = self.data[indices.data[i]]
