cimport numpy as np

from ..containers.containers cimport CarrayContainer
from .gravity_pool cimport Node


cdef class Splitter:

    cdef int dim  # dimension of simulation
    cdef long idp # particle id to process

    cdef void initialize_particles(self, CarrayContainer particles)
    cdef void process_particle(self, long idp)
    cdef int split(self, Node* node)

cdef class BarnesHut(Splitter):

    cdef double open_angle
    cdef np.float64_t *x[3]
