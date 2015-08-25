from utils.carray cimport DoubleArray, LongLongArray
from particles.particle_array cimport ParticleArray
from reconstruction.reconstruction cimport ReconstructionBase


cdef class FluxBase:

    cdef public object mesh
    cdef public ReconstructionBase reconstruction
    cdef public double gamma
    cdef public double cfl

    cdef solve(self, DoubleArray fluxes, DoubleArray left_face, DoubleArray right_face, DoubleArray faces,
            double t, double dt, int iteration_count)

cdef class HLLC(FluxBase):

    cdef void get_waves(self, double d_l, double u_l, double p_l,
            double d_r, double u_r, double p_r,
            double gamma, double *sl, double *sc, double *sr)
