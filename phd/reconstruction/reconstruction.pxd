cimport numpy as np

from ..mesh.mesh cimport Mesh
from ..riemann.riemann cimport RiemannBase
from ..domain.domain_manager cimport DomainManager
from ..containers.containers cimport CarrayContainer
from ..equation_state.equation_state cimport EquationStateBase

cdef class ReconstructionBase:

    cdef bint fields_registered
    cdef bint has_passive_scalars

    cdef public CarrayContainer left_states
    cdef public CarrayContainer right_states

    cdef dict reconstruct_fields
    cdef dict reconstruct_field_groups

    cdef int num_passive
    cdef np.float64_t** passive
    cdef np.float64_t** passive_l
    cdef np.float64_t** passive_r
    cdef np.float64_t** dpassive

    cpdef compute_gradients(self, CarrayContainer particles, Mesh mesh,
                            DomainManager domain_manager)

    cpdef add_spatial(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost)

    cpdef add_temporal(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost)

    cpdef compute_states(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost)

cdef class PieceWiseConstant(ReconstructionBase):
    pass

cdef class PieceWiseLinear(ReconstructionBase):

    cdef public int limiter

    cdef public CarrayContainer grad

    cdef dict reconstruct_grads
    cdef dict reconstruct_grad_groups

    cdef np.float64_t** prim_pointer
    cdef np.float64_t** grad_pointer

    cdef np.float64_t* phi_max
    cdef np.float64_t* phi_min

    cdef np.float64_t* alpha
    cdef np.float64_t* df
