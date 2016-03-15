from containers.containers cimport CarrayContainer
from reconstruction.reconstruction cimport ReconstructionBase


cdef class RiemannBase:

    cdef public ReconstructionBase reconstruction
    cdef public double gamma
    cdef public double cfl

    cdef _solve(self, CarrayContainer fluxes, CarrayContainer left_face, CarrayContainer right_face, CarrayContainer faces,
            double t, double dt, int iteration_count, int dim)

cdef class HLL(RiemannBase):

    cdef void get_waves(self, double d_l, double u_l, double p_l,
            double d_r, double u_r, double p_r,
            double gamma, double *sl, double *sc, double *sr)

#cdef class HLLC(HLL):
#    pass
