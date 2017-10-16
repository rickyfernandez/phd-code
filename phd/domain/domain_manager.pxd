cimport numpy as np
from libcpp.vector cimport vector
from libcpp.list cimport list as cpplist

from ..domain.domain cimport DomainLimits
from ..domain.boundary cimport BoundaryConditionBase
from ..containers.containers cimport CarrayContainer
from ..load_balance.load_balance cimport LoadBalance
from ..utils.carray cimport DoubleArray, LongLongArray, LongArray, IntArray


cdef extern from "particle.h":
    cdef struct FlagParticle:
        double x[3]
        double v[3]
        int index
        double radius
        double search_radius

    cdef cppclass BoundaryParticle:
        BoundaryParticle(double _x[3], double _v[3], int _index, int _proc,
                int _boundary_type, int dim)
        double x[3]
        double v[3]
        int proc
        int index
        int boundary_type

    FlagParticle* particle_flag_deref(cpplist[FlagParticle].iterator &it)

cdef class DomainManager:

    cdef public DomainLimits domain
    cdef public DoubleArray old_radius
    cdef public LoadBalance load_balance
    cdef public BoundaryConditionBase boundary_condition

    cdef public double param_initial_radius
    cdef public double param_box_fraction
    cdef public double param_search_radius_factor

    # hold/flag particle for ghost creation 
    cdef vector[BoundaryParticle] ghost_vec
    cdef cpplist[FlagParticle] flagged_particles

    # for parallel runs
    cdef public np.ndarray send_cnts    # send counts for mpi
    cdef public np.ndarray recv_cnts    # send counts for mpi
    cdef public np.ndarray send_disp    # send displacments for mpi
    cdef public np.ndarray recv_disp    # receive displacments for mpi

    # load balance methods
    cpdef check_for_partition(self, CarrayContainer particles)
    cpdef partition(self, CarrayContainer particles)

    # ghost generation
    #cdef filter_radius(self, CarrayContainer particles)
    cpdef setup_for_ghost_creation(self, CarrayContainer particles)

    cdef create_ghost_particles(self, CarrayContainer particles)
    cdef create_interior_ghost_particle(self, FlagParticle* p)

    cdef update_search_radius(self, CarrayContainer particles)

    cdef copy_particles(self, CarrayContainer particles)
    cdef copy_particles_serial(self, CarrayContainer particles)
    cdef copy_particles_parallel(self, CarrayContainer particles)

    cdef bint ghost_complete(self)
    cdef values_to_ghost(self, CarrayContainer particles, list fields)
