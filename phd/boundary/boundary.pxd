
from domain.domain cimport DomainLimits
from containers.containers cimport ParticleContainer

cdef class BoundaryBase2d:
    cdef public DomainLimits domain

    cdef int _create_ghost_particles(ParticleContainer pc)

cdef class Reflect2d(BoundaryBase2d):
    pass
