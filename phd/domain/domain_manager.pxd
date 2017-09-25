
cimport numpy as np
from libcpp.vector cimport vector

from ..domain.domain cimport DomainLimits
from ..utils.carray cimport DoubleArray, LongLongArray, LongArray, IntArray


cdef extern from "particle.h":
    cdef cppclass QueryParticle:
        Particle(double _x[3], double _v[3], double _old_radius, double _new_radius, int dim)
        double x[3]
        double v[3]
        double old_radius
        double new_radius
        int index

    cdef cppclass BoundaryParticle:
        Particle(double _x[3], double _v[3], int _index, int _proc, int dim)
        double x[3]
        double v[3]
        int proc
        int index

cdef class DomainManager:

    cdef public DomainLimits domain
    cdef public DoubleArray old_radius
    cdef public LoadBalance load_balance
    cdef public BoundaryConditionBase boundary_condition

    cdef public double param_box_fraction

    # flag interior particles
    cdef vector[int] old_interior_flagged
    cdef vector[int] new_interior_flagged

    # flag exterior particles
    cdef vector[int] old_exterior_flagged
    cdef vector[int] new_exterior_flagged

    # load balance methods
    cpdef check_for_partition(self, CarrayContainer particles)
    cpdef partition(self, CarrayContainer particles)

    # ghost generation
    cdef filter_radius(self, CarrayContainer particles)
    cdef create_ghost_particles(CarrayContainer particles)

    cdef copy_particles(CarrayContainer particles, vector[BoundaryParticle] ghost_vec)
    cdef copy_particles_serial(CarrayContainer particles, vector[BoundaryParticle] ghost_vec)
    cdef copy_particles_parallel(CarrayContainer particles, vector[BoundaryParticle] ghost_vec)

    cdef bint ghost_complete(self)

    # domain querying
