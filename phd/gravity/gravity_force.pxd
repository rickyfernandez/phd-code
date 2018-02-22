from ..source_term.source_term cimport MUSCLHancockSourceTerm

cdef class ConstantGravity(MUSCLHancockSourceTerm):

    cdef double g
    cdef int axis
    cdef str grav_axis
