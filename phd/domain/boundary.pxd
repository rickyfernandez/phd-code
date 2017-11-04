cimport numpy as np

from ..domain.domain_manager cimport DomainManager
from ..containers.containers cimport CarrayContainer
from ..domain.domain_manager cimport FlagParticle, BoundaryParticle

cdef enum:
    REFLECTIVE = 0x01
    PERIODIC   = 0x02

cdef inline bint in_box(double x[3], double r, np.float64_t bounds[2][3], int dim)

cdef class BoundaryConditionBase:
    cdef void create_ghost_particle(self, FlagParticle *p, DomainManager domain_manager)
    cdef void create_ghost_particle_serial(self, FlagParticle *p, DomainManager domain_manager)
    cdef void create_ghost_particle_parallel(self, FlagParticle *p, DomainManager domain_manager)
    cdef void migrate_particles(self, CarrayContainer particles, DomainManager domain_manager)

cdef class Reflective(BoundaryConditionBase):
    pass

cdef class Periodic(BoundaryConditionBase):
    pass
