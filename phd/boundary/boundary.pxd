
from domain.domain cimport DomainLimits
from containers.containers cimport ParticleContainer

cdef class BoundaryBase:
    cdef public DomainLimits domain

    cdef int _create_ghost_particles(self, ParticleContainer pc)

cdef class Reflect2d(BoundaryBase):
    pass

cdef class Reflect3d(BoundaryBase):
    pass
