cimport numpy as np

cdef class DomainLimits:

    #cdef public np.float64_t[3] xmin
    #cdef public np.float64_t[3] xmax
    #cdef public np.float64_t[3] xtranslate
    cdef np.float64_t[3] translate
    cdef np.float64_t[2][3] bounds
    cdef public np.float64_t max_length, min_length

    cdef public int dim
    cdef public bint is_periodic
    cdef public bint is_outflow
    cdef public bint is_wall

    cdef _check_limits(self, xmin, xmax)
