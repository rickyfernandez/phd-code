cimport numpy as np

from .splitter cimport Splitter
from .gravity_pool cimport Node
from ..containers.containers cimport CarrayContainer


cdef class Interaction:
    cdef int dim
    cdef long current
    cdef long num_particles

    cdef IntArray tags
    cdef Splitter splitter

    cdef void interact(self, Node* node)
    cdef void initialize_particles(self, CarrayContainer pc)
    cdef int process_particle(self)
    cpdef void set_splitter(self, Splitter splitter)

cdef class GravityAcceleration(Interaction):
    cdef np.float64_t *x[3]
    cdef np.float64_t *a[3]
