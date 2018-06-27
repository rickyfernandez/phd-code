from ..source_term.source_term cimport MUSCLHancockSourceTerm
from .gravity_tree cimport GravityTree

cdef class ConstantGravity(MUSCLHancockSourceTerm):

    cdef double g
    cdef int axis
    cdef str grav_axis

cdef class SelfGravity(MUSCLHancockSourceTerm):

    cdef public double eta
    cdef public str split_type
    cdef public double barnes_angle
    cdef public int max_buffer_size
    cdef public double smoothing_length
    cdef public int calculate_potential
    cdef GravityTree gravity 
