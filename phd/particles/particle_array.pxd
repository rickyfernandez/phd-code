cimport numpy as np
from ..utils.carray cimport BaseArray

cdef class ParticleArray:

    cdef readonly long num_real_particles
    cdef readonly long num_ghost_particles

    cdef readonly bint is_dirty
    cdef readonly bint indices_invalid

    cdef readonly dict properties
    cdef readonly list field_names

    cpdef int get_number_of_particles(self)
    cpdef remove_particles(self, np.ndarray index_list)
    cpdef remove_tagged_particles(self, np.int8_t tag)
    cpdef extend(self, int num_particles)
    cpdef BaseArray get_carray(self, str prop)
    cdef  _check_property(self, str prop)
    cpdef int align_particles(self) except -1
    cpdef resize(self, long size)
    cdef void make_ghost(self, np.float64_t x, np.float64_t y, np.int32_t proc)
