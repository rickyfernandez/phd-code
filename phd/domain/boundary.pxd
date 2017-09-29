cimport numpy as np

from ..domain.domain_manager cimport DomainManager
from ..domain.domain_manager cimport FlagParticle, BoundaryParticle


cdef int in_box(np.float64_t x[3], np.float64_t r, np.float64_t bounds[2][3], int dim)

cdef class BoundaryBase:
    cdef void create_ghost_particle(FlagParticle *p, DomainManager domain_manager)
    cdef void create_ghost_particle_serial(FlagParticle *p, DomainManager domain_manager):
    cdef void create_ghost_particle_parallel(FlagParticle *p, DomainManager domain_manager):

cdef class Reflective(BoundaryConditionBase):
    pass

cdef class Periodic(BoundaryConditionBase):
    pass
