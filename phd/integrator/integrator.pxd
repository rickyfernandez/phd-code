
from particles.particle_array cimport ParticleArray
from right_hand_side.right_hand_side cimport RightHandSide


cdef class Integrator:
    ############################################################
    # Data attributes
    ############################################################
    cdef public ParticleArray pa
    cdef public ParticleArray left_state
    cdef public ParticleArray right_state
    cdef public ParticleArray fluxes

    cdef public RightHandSide rhs
    cdef public object mesh

    # particle properties used for time stepping
    cdef int regularize
    cdef double eta

    ############################################################
    # Member functions
    ############################################################
    cdef _integrate(self, double dt, double t, int iteration_count)
    cdef _compute_face_velocities(self)
    cdef _assign_particle_velocities(self)
