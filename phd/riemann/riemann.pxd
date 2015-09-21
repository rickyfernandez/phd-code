from particles.particle_array cimport ParticleArray
from reconstruction.reconstruction cimport ReconstructionBase


cdef class RiemannBase:

    cdef public object mesh
    cdef public ReconstructionBase reconstruction
    cdef public double gamma
    cdef public double cfl

    cdef solve(self, ParticleArray fluxes, ParticleArray left_face, ParticleArray right_face, ParticleArray faces,
            double t, double dt, int iteration_count)

cdef class HLLC(RiemannBase):

    cdef void get_waves(self, double d_l, double u_l, double p_l,
            double d_r, double u_r, double p_r,
            double gamma, double *sl, double *sc, double *sr)
