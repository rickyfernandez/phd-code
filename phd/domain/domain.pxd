
cdef class DomainLimits:

    cdef public double xmin, xmax
    cdef public double ymin, ymax
    cdef public double zmin, zmax

    cdef public double xtranslate
    cdef public double ytranslate
    cdef public double ztranslate

    cdef public int dim
    cdef public bint is_periodic
    cdef public bint is_outflow
    cdef public bint is_wall

    cdef _check_limits(self, xmin, xmax)
