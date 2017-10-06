cimport numpy as np

from ..domain.domain_manager cimport DomainManager
from ..domain.domain_manager cimport FlagParticle, BoundaryParticle

cdef enum:
    REFLECTIVE = 0x01
    PERIODIC   = 0x02

cdef int in_box(double x[3], double r, np.float64_t bounds[2][3], int dim)

cdef class BoundaryBase:
    cdef void create_ghost_particle(FlagParticle *p, DomainManager domain_manager)
    cdef void create_ghost_particle_serial(FlagParticle *p, DomainManager domain_manager):
    cdef void create_ghost_particle_parallel(FlagParticle *p, DomainManager domain_manager):

cdef class Reflective(BoundaryConditionBase):
    pass

cdef class Periodic(BoundaryConditionBase):
    pass
