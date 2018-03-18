from ..source_term.source_term cimport MUSCLHancockSourceTerm
from .gravity_tree cimport GravityTree

cdef class ConstantGravity(MUSCLHancockSourceTerm):

    cdef double g
    cdef int axis
    cdef str grav_axis

cdef class SelfGravity(MUSCLHancockSourceTerm):

    cdef double eta
    cdef GravityTree gravity 
