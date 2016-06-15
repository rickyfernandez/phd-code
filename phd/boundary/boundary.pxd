
cimport numpy as np
from utils.carray cimport LongArray
from load_balance.tree cimport Tree
from domain.domain cimport DomainLimits
from load_balance.load_balance cimport LoadBalance
from containers.containers cimport ParticleContainer, CarrayContainer

cdef extern from "particle.h":
    cdef cppclass Particle:
        Particle(double _x[3], double _v[3], int dim)
        double x[3]
        double v[3]

cdef int in_box(np.float64_t x[3], np.float64_t r, np.float64_t bounds[2][3], int dim)
cdef _reflective(ParticleContainer pc, DomainLimits domain, int num_real_particles)
cdef _periodic(ParticleContainer pc, DomainLimits domain, int num_real_particles)
cdef _periodic_parallel(ParticleContainer pc, CarrayContainer ghost, DomainLimits domain,
        Tree glb_tree, np.ndarray leaf_npy, LongArray buffer_ids, LongArray buffer_pid,
        int num_real_particles, int rank)

cdef class Boundary:
    cdef public DomainLimits domain
    cdef public int boundary_type

    cdef int start_ghost

    cdef np.ndarray send_particles, recv_particles

    cdef _set_radius(self, ParticleContainer pc, int num_real_particles)
    cdef int _create_ghost_particles(self, ParticleContainer pc)
    cdef _update_ghost_particles(self, ParticleContainer pc, dict fields)

cdef class BoundaryParallel(Boundary):

    cdef public LoadBalance load_bal

    cdef public object comm
    cdef public int rank, size, dim

    cdef public LongArray buffer_ids
    cdef public LongArray buffer_pid

    cdef CarrayContainer _create_interior_ghost_particles(self, ParticleContainer pc, int num_real_particles)
    cdef migrate_boundary_particles(self, ParticleContainer pc)
