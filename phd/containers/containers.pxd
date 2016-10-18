cimport numpy as np
from ..utils.carray cimport BaseArray


cdef class CarrayContainer:

    cdef readonly dict properties
    cdef readonly dict carray_info
    cdef readonly dict named_groups

    cpdef int get_number_of_items(self)
    cpdef remove_items(self, np.ndarray index_list)
    cpdef extend(self, int num_particles)
    cdef void pointer_groups(self, np.float64_t *vec[], list field_names)
    cpdef BaseArray get_carray(self, str prop)
    cdef  _check_property(self, str prop)
    cpdef resize(self, int size)
    cpdef remove_tagged_particles(self, np.int8_t tag)
    cpdef CarrayContainer extract_items(self, np.ndarray index_array, list fields=*)
    cpdef int append_container(self, CarrayContainer carray)
