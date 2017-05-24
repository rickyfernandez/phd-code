
cimport numpy as np

# forward declaration
cdef class BaseArray

cdef class BaseArrayIter:
    cdef BaseArray arr
    cdef int i

cdef class BaseArray:
    """Base Class for managed C-arrays."""
    cdef readonly long length, alloc
    cdef np.ndarray _npy_array

    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef np.ndarray get_npy_array(self)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)
    cpdef reset(self)
    cpdef shrink(self, long size)

    cpdef align_array(self, np.ndarray new_indices)
    cpdef str get_c_type(self)
    cpdef copy_values(self, LongArray indices, BaseArray dest)
    cpdef paste_values(self, LongArray indices, BaseArray dest)
    cpdef add_values(self, LongArray indices, BaseArray dest)

cdef class DoubleArray(BaseArray):
    """This class defines a managed array of np.float64_t"""
    cdef np.float64_t *data
    cdef readonly np.float64_t minimum, maximum

    cdef _setup_npy_array(self)
    cdef np.float64_t* get_data_ptr(self)

    cpdef np.float64_t get(self, long pid)
    cpdef set(self, long pid, np.float64_t value)
    cpdef append(self, np.float64_t value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)

    cpdef align_array(self, np.ndarray new_indices)
    cpdef str get_c_type(self)
    cpdef copy_values(self, LongArray indices, BaseArray dest)
    cpdef paste_values(self, LongArray indices, BaseArray dest)
    cpdef add_values(self, LongArray indices, BaseArray dest)

cdef class IntArray(BaseArray):
    """This class defines a managed array of np.int8_t"""
    cdef np.int8_t *data
    cdef readonly np.int8_t minimum, maximum

    cdef _setup_npy_array(self)
    cdef np.int8_t* get_data_ptr(self)

    cpdef np.int8_t get(self, long pid)
    cpdef set(self, long pid, np.int8_t value)
    cpdef append(self, np.int8_t value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)

    cpdef align_array(self, np.ndarray new_indices)
    cpdef str get_c_type(self)
    cpdef copy_values(self, LongArray indices, BaseArray dest)
    cpdef paste_values(self, LongArray indices, BaseArray dest)
    cpdef add_values(self, LongArray indices, BaseArray dest)

cdef class LongArray(BaseArray):
    """This class defines a managed array of np.int32_t"""
    cdef np.int32_t *data
    cdef readonly np.int32_t minimum, maximum

    cdef _setup_npy_array(self)
    cdef np.int32_t* get_data_ptr(self)

    cpdef np.int32_t get(self, long pid)
    cpdef set(self, long pid, np.int32_t value)
    cpdef append(self, np.int32_t value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)

    cpdef align_array(self, np.ndarray new_indices)
    cpdef str get_c_type(self)
    cpdef copy_values(self, LongArray indices, BaseArray dest)
    cpdef paste_values(self, LongArray indices, BaseArray dest)
    cpdef add_values(self, LongArray indices, BaseArray dest)

cdef class LongLongArray(BaseArray):
    """This class defines a managed array of np.int64_t"""
    cdef np.int64_t *data
    cdef readonly np.int64_t minimum, maximum

    cdef _setup_npy_array(self)
    cdef np.int64_t* get_data_ptr(self)

    cpdef np.int64_t get(self, long pid)
    cpdef set(self, long pid, np.int64_t value)
    cpdef append(self, np.int64_t value)
    cpdef reserve(self, long size)
    cpdef resize(self, long size)
    cpdef squeeze(self)
    cpdef remove(self, np.ndarray index_list, bint input_sorted=*)
    cpdef extend(self, np.ndarray in_array)

    cpdef align_array(self, np.ndarray new_indices)
    cpdef str get_c_type(self)
    cpdef copy_values(self, LongArray indices, BaseArray dest)
    cpdef paste_values(self, LongArray indices, BaseArray dest)
    cpdef add_values(self, LongArray indices, BaseArray dest)
