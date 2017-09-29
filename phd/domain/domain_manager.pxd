cimport numpy as np
from libcpp.list cimport list
from libcpp.vector cimport vector

from ..domain.domain cimport DomainLimits
from ..utils.carray cimport DoubleArray, LongLongArray, LongArray, IntArray


cdef extern from "particle.h":
    cdef struct FlagParticle:
        double x[3]
        double v[3]
        int index
        double old_radius
        double new_radius
        int boundary_type

    cdef cppclass BoundaryParticle:
        Particle(double _x[3], double _v[3], int _index, int _proc, int dim)
        double x[3]
        double v[3]
        int proc
        int index

    FlagParticle* particle_flag_deref(list[FlagParticle].iterator &it)

cdef class DomainManager:

    cdef public DomainLimits domain
    cdef public DoubleArray old_radius
    cdef public LoadBalance load_balance
    cdef public BoundaryConditionBase boundary_condition

    cdef public double param_box_fraction
    cdef public double param_search_radius_factor

    # flag interior particles
    cdef list[FlagParticle] flagged_particles

    # for parallel runs
    cdef public np.ndarray send_cnts      # send counts for mpi
    cdef public np.ndarray recv_cnts      # send counts for mpi
    cdef public np.ndarray send_disp      # send displacments for mpi
    cdef public np.ndarray recv_disp      # receive displacments for mpi

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
