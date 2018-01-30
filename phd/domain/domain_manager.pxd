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
        double old_search_radius
        double search_radius

    cdef cppclass BoundaryParticle:
        BoundaryParticle(double _x[3], double _v[3], int _index,
                int _proc, int dim)
        double x[3]
        double v[3]
        int proc
        int index

    cdef cppclass GhostID:
        GhostID(int _index, int _proc, int _export_num)
        int index;
        int proc;
        int export_num;

    FlagParticle* particle_flag_deref(cpplist[FlagParticle].iterator &it)


cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)


cdef bint boundary_particle_cmp(
        const BoundaryParticle &a, const BoundaryParticle &b) nogil

cdef bint ghostid_cmp(
        const GhostID &a, const GhostID &b) nogil

cdef class DomainManager:

    cdef public DomainLimits domain
    cdef public DoubleArray old_radius
    cdef public LoadBalance load_balance
    cdef public BoundaryConditionBase boundary_condition

    cdef public double initial_radius
    cdef public double search_radius_factor

    cdef bint particle_fields_registered

    # hold/flag particle for ghost creation 
    cdef vector[BoundaryParticle] ghost_vec
    cdef cpplist[FlagParticle] flagged_particles

    # for parallel runs
    cdef int num_export

    cdef np.ndarray loc_done
    cdef np.ndarray glb_done

    cdef vector[GhostID] export_ghost_buffer
    cdef vector[GhostID] import_ghost_buffer

    cdef np.ndarray send_cnts    # send counts for mpi
    cdef np.ndarray recv_cnts    # send counts for mpi
    cdef np.ndarray send_disp    # send displacments for mpi
    cdef np.ndarray recv_disp    # receive displacments for mpi

    # load balance methods
    cpdef check_for_partition(self, CarrayContainer particles)
    cpdef partition(self, CarrayContainer particles)

    # ghost generation
    #cdef filter_radius(self, CarrayContainer particles)
    cpdef setup_initial_radius(self, CarrayContainer particles)
    cpdef setup_for_ghost_creation(self, CarrayContainer particles)

    cpdef create_ghost_particles(self, CarrayContainer particles)
    cdef create_interior_ghost_particles(self, CarrayContainer particles)

    cpdef update_search_radius(self, CarrayContainer particles)

    cdef copy_particles_serial(self, CarrayContainer particles)
    cdef copy_particles_parallel(self, CarrayContainer particles)

    cpdef move_generators(self, CarrayContainer particles, double dt)
    cpdef migrate_particles(self, CarrayContainer particles)

    cpdef bint ghost_complete(self)

    cdef update_ghost_fields(self, CarrayContainer particles, list fields)
    cdef update_ghost_gradients(self, CarrayContainer particles, CarrayContainer gradients)
    cdef reindex_ghost(self, CarrayContainer particles, int num_real_particles,
                       int total_num_particles)
