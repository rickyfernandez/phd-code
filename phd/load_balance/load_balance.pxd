import numpy as np
cimport numpy as np

from .tree cimport Tree, hilbert_type
from utils.carray cimport LongArray, LongLongArray
from containers.containers cimport ParticleContainer


cdef class LoadBalance:

    cdef public object comm
    cdef public np.int32_t rank
    cdef public np.int32_t size

    cdef public np.int32_t order

    cdef public np.float64_t fac
    cdef public np.ndarray corner
    cdef public np.float64_t factor
    cdef public np.float64_t box_length

    cdef public export_ids
    cdef public export_pid

    cdef public Tree tree
    cdef public LongArray leaf_pid

    cdef hilbert_type hilbert_func

    cdef void _calculate_local_work(self, ParticleContainer pc, np.ndarray work)
    cdef void _find_split_in_work(self, np.ndarray global_work)
    cdef void _collect_particles_export(self, ParticleContainer pc, LongArray part_ids, LongArray part_pid,
            LongArray leaf_pid, int my_pid)
    cdef void _compute_hilbert_keys(self, ParticleContainer pc)
