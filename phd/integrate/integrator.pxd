from containers.containers cimport CarrayContainer, ParticleContainer
from riemann.riemann cimport RiemannBase

cdef class IntegrateBase:

    cdef public ParticleContainer particles
    cdef public CarrayContainer left_state
    cdef public CarrayContainer right_state
    cdef public CarrayContainer flux

    cdef public double gamma

    cdef public RiemannBase riemann
    cdef public object mesh

    cdef double _compute_time_step(self)
    cdef _integrate(self, double dt, double t, int iteration_count)

cdef class MovingMesh(IntegrateBase):

    # particle properties used for time stepping
    cdef int regularize
    cdef double eta

    cdef _integrate(self, double dt, double t, int iteration_count)
    cdef _compute_face_velocities(self)
    cdef _assign_particle_velocities(self)
    cdef _assign_face_velocities(self)
