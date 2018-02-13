import numpy as np
cimport numpy as np

from ..load_balance.tree cimport Tree, hilbert_type
from ..utils.carray cimport LongArray, LongLongArray
from ..containers.containers cimport CarrayContainer


cdef class LoadBalance:

    cdef public np.int32_t order
    cdef public np.float64_t factor
    cdef public np.int32_t min_in_leaf

    cdef int dim
    cdef np.float64_t fac
    cdef double[2][3] bounds

    cdef bint domain_info_added

    cdef double corner[3]
    cdef np.float64_t box_length

    cdef export_ids
    cdef export_pid

    cdef Tree tree
    cdef LongArray leaf_pid

    cdef hilbert_type hilbert_func

    cdef void calculate_local_work(self, CarrayContainer particles, np.ndarray work)
    cdef void find_split_in_work(self, np.ndarray global_work)
    cdef void collect_particles_export(self, CarrayContainer particles, LongArray part_ids, LongArray part_pid,
            LongArray leaf_pid, int my_pid)
    cdef void compute_hilbert_keys(self, CarrayContainer particles)
