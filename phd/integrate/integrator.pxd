from particles.particle_array cimport ParticleArray
from riemann.riemann cimport RiemannBase

cdef class IntegrateBase:

    cdef public ParticleArray pa
    cdef public ParticleArray left_state
    cdef public ParticleArray right_state
    cdef public ParticleArray flux

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
