from ..containers.containers cimport CarrayContainer
from ..reconstruction.reconstruction cimport ReconstructionBase


cdef class RiemannBase:

    cdef public ReconstructionBase reconstruction

    cdef public double cfl
    cdef public double gamma
    cdef public int boost
    cdef public int dim

    cdef _solve(self, CarrayContainer fluxes, CarrayContainer left_face, CarrayContainer right_face, CarrayContainer faces,
            double t, double dt, int iteration_count, int dim)

    cdef _deboost(self, CarrayContainer fluxes, CarrayContainer faces, int dim)

    cdef double _compute_time_step(self, CarrayContainer pc)

cdef class HLL(RiemannBase):

    cdef inline void get_waves(self, double dl, double ul, double pl,
            double dr, double ur, double pr,
            double gamma, double *sl, double *sc, double *sr)

cdef class HLLC(HLL):
    pass

cdef class Exact(RiemannBase):

    cdef inline double p_guess(self, double dl, double ul, double pl, double cl,
            double dr, double ur, double pr, double cr, double gamma) nogil

    cdef inline double p_func(self, double d, double u, double p,
            double c, double gamma, double p_old) nogil

    cdef inline double p_func_deriv(self, double d, double u, double p,
            double c, double gamma, double p_old) nogil

    cdef inline double get_pstar(self, double dl, double ul, double pl, double cl,
            double dr, double ur, double pr, double cr, double gamma) nogil

    cdef inline void vacuum(self,
            double dl, double vl[3], double pl, double vnl, double cl,
            double dr, double vr[3], double pr, double vnr, double cr,
            double *d, double  v[3], double *p, double *vn, double *vsq,
            double gamma, double n[3], int dim) nogil
