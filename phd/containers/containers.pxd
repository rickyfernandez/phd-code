cimport numpy as np
from ..utils.carray cimport BaseArray, LongArray


cdef class CarrayContainer:

    cdef readonly dict carrays
    cdef readonly dict carray_dtypes
    cdef readonly dict carray_named_groups

    cpdef register_carray(self, int carray_size, str carray_name, str dtype=*)

    cpdef int get_carray_size(self)
    cpdef remove_items(self, np.ndarray index_list)
    cpdef extend(self, int increase_carray_size)
    cdef void pointer_groups(self, np.float64_t *vec[], list carray_list_names)
    cpdef BaseArray get_carray(self, str carray_name)
    cpdef resize(self, int carray_size)
    cpdef remove_tagged_particles(self, np.int8_t tag)
    cpdef CarrayContainer extract_items(self, LongArray index_array, list carray_list_names=*)
    cpdef int append_container(self, CarrayContainer container)
    cpdef copy(self, CarrayContainer container, LongArray indices, list carray_list_names)
    cpdef paste(self, CarrayContainer container, LongArray indices, list carray_list_names)
    cpdef add(self, CarrayContainer container, LongArray indices, list carray_list_names)
