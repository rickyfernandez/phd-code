cimport numpy as np

from ..mesh.mesh cimport Mesh
from ..riemann.riemann cimport RiemannBase
from ..domain.domain_manager cimport DomainManager
from ..containers.containers cimport CarrayContainer
from ..equation_state.equation_state cimport EquationStateBase

cdef class ReconstructionBase:

#    cdef bint do_colors
#    cdef np.float64_t** col
#    cdef np.float64_t** coll
#    cdef np.float64_t** colr

    cdef bint registered_fields
    cdef public CarrayContainer left_states
    cdef public CarrayContainer right_states

    cdef public dict reconstruct_fields
    cdef public dict reconstruct_field_groups
#    cdef public dict reconstruct_grad_groups

    cpdef compute_states(self, CarrayContainer particles, Mesh mesh,
            bint boost, DomainManager domain_manager, double dt)

cdef class PieceWiseConstant(ReconstructionBase):
    pass

#cdef class PieceWiseLinear(ReconstructionBase):
#
#    cdef public int param_limiter
#    cdef public CarrayContainer grad
#
#    cdef np.float64_t** prim_ptr
#    cdef np.float64_t** grad_ptr
#
#    cdef np.float64_t* phi_max
#    cdef np.float64_t* phi_min
#
#    cdef np.float64_t* alpha
#    cdef np.float64_t* df
#
#    cdef compute_gradients(self, CarrayContainer pc, CarrayContainer faces, Mesh mesh)
