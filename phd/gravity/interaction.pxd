cimport numpy as np

from .splitter cimport Splitter
from .gravity_pool cimport Node
from ..utils.carray cimport IntArray
from ..containers.containers cimport CarrayContainer

# ** later on this will become a generic class to hanlde
#    any type of calculation on the tree  **
cdef class Interaction:
    cdef int dim                    # spatial dimension of the problem
    cdef long current               # index of particle to compute on
    cdef long current_node          # current node in tree walk for particle
    cdef long num_particles         # total number of particles
    cdef bint particle_done         # flag to know particle is done walking
    cdef bint local_particles       # flag indicating local or imported particles

    cdef public dict fields         # fields to use in computation
    cdef public dict carray_named_groups   # vector of fields for ease

    cdef IntArray tags              # reference to particle tags
    cdef Splitter splitter          # criteria to open node

    # compute calculation between particle and node
    cdef void initialize_particles(self, CarrayContainer pc, bint local_particles=*)
    cdef void interact(self, Node* node)

    # methods to check status or flag
    cdef bint process_particle(self)
    cdef bint done_processing(self)
    cdef void particle_finished(self)

    # methods to hault/start walk
    cdef void particle_not_finished(self, long node_index)
    cdef long start_node_index(self)

cdef class GravityAcceleration(Interaction):
    cdef int calc_potential         # flag to include gravity potential
    cdef double smoothing_length    # gravitational smoothing length

    # pointer to particle data
    cdef np.float64_t *pot          # gravitational potential
    cdef np.float64_t *x[3]         # positions
    cdef np.float64_t *a[3]         # accelerations
