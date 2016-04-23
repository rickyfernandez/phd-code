
cimport numpy as np
from domain.domain cimport DomainLimits
from containers.containers cimport ParticleContainer
from utils.carray cimport LongArray

cdef extern from "particle.h":
    cdef cppclass Particle:
        Particle(double _x[3], double _v[3], int dim)
        double x[3]
        double v[3]

cdef class BoundaryBase:
    cdef public DomainLimits domain

    cdef int _create_ghost_particles(self, ParticleContainer pc)
    cdef _update_ghost_particles(self, ParticleContainer pc)

cdef class Reflect2d(BoundaryBase):
    pass

cdef class Reflect3d(BoundaryBase):
    pass

cdef class BoundaryParallelBase:

    cdef public object load_bal

    cdef public object comm
    cdef public int rank, size, dim

    cdef public LongArray buffer_ids
    cdef public LongArray buffer_pid

    cdef np.float64_t bounds[2][3]

    cdef _interior_ghost_particles(self, ParticleContainer pc)
    cdef _exterior_ghost_particles(self, ParticleContainer pc)
    cdef int _create_ghost_particles(self, ParticleContainer pc)
    cdef _send_ghost_particles(self, ParticleContainer pc)

#cdef class ReflectParalle(BoundaryParallelBase):
#    pass
