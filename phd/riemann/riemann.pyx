from ..containers.containers cimport CarrayContainer
from ..utils.carray cimport DoubleArray
from ..reconstruction.reconstruction cimport ReconstructionBase

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, pow, fmin, fmax

cdef class RiemannBase:
    def __init__(self, double gamma=1.4, double cfl=0.3, **kwargs):
        # self.reconstruction = None
        self.gamma = gamma
        self.cfl = cfl

    def solve(self, fluxes, left_faces, right_faces, faces, t, dt, iteration_count, dim):
        self._solve(fluxes, left_faces, right_faces, faces, t, dt, iteration_count, dim)

    cdef _solve(self, CarrayContainer fluxes, CarrayContainer left_faces, CarrayContainer right_faces, CarrayContainer faces,
            double t, double dt, int iteration_count, int dim):
        msg = "RiemannBase::solve called!"
        raise NotImplementedError(msg)

cdef class HLL(RiemannBase):

    cdef _solve(self, CarrayContainer fluxes, CarrayContainer left_faces, CarrayContainer right_faces, CarrayContainer faces,
            double t, double dt, int iteration_count, int dim):

        # left state primitive variables
        cdef DoubleArray dl = left_faces.get_carray("density")
        cdef DoubleArray pl = left_faces.get_carray("pressure")

        # left state primitive variables
        cdef DoubleArray dr = right_faces.get_carray("density")
        cdef DoubleArray pr = right_faces.get_carray("pressure")

        cdef DoubleArray fm  = fluxes.get_carray("mass")
        cdef DoubleArray fe  = fluxes.get_carray("energy")

        # local variables
        cdef int i, k
        cdef double fac1, fac2, el, er
        cdef double _dl, _pl
        cdef double _dr, _pr
        cdef double wn, Vnl, Vnr, sl, sr, s_contact
        cdef double vl_tmp, vr_tmp, nx_tmp, vl_sq, vr_sq
        cdef np.float64_t *vl[3], *vr[3], *fmv[3], *nx[3], *wx[3]

        cdef double gamma = self.gamma
        cdef int num_faces = faces.get_number_of_items()

        left_faces.pointer_groups(vl,  left_faces.named_groups['velocity'])
        right_faces.pointer_groups(vr, right_faces.named_groups['velocity'])
        fluxes.pointer_groups(fmv, fluxes.named_groups['momentum'])
        faces.pointer_groups(nx, faces.named_groups['normal'])
        faces.pointer_groups(wx, faces.named_groups['velocity'])

        for i in range(num_faces):

            # left state
            _dl = dl.data[i]
            _pl = pl.data[i]

            # right state
            _dr = dr.data[i]
            _pr = pr.data[i]

            Vnl = Vnr = 0.0
            vl_sq = vr_sq = wn = 0.0
            for k in range(dim):

                vl_tmp = vl[k][i]; vr_tmp = vr[k][i]
                nx_tmp = nx[k][i]

                # left/right velocity square
                vl_sq += vl_tmp*vl_tmp
                vr_sq += vr_tmp*vr_tmp

                # project left/righ velocity to face normal
                Vnl += vl_tmp*nx_tmp
                Vnr += vr_tmp*nx_tmp

                # project face velocity to face normal
                wn += wx[k][i]*nx_tmp

            self.get_waves(_dl, Vnl, _pl, _dr, Vnr, _pr, gamma,
                    &sl, &s_contact, &sr)

            # calculate interface flux - eq. 10.21
            if(wn <= sl):

                # left state
                fm.data[i]  = _dl*(Vnl - wn)
                fe.data[i]  = (0.5*_dl*vl_sq + _pl/(gamma - 1.0))*(Vnl - wn) + _pl*Vnl

                for k in range(dim):
                    fmv[k][i] = _dl*vl[k][i]*(Vnl - wn) + _pl*nx[k][i]

            elif((sl < wn) and (wn <= sr)):

                fac1 = sr - wn
                fac2 = sl - wn
                fac3 = sr - sl

                # eqs. 10.20 and 10.13
                fm.data[i] = (_dl*Vnl*fac1 - _dr*Vnr*fac2 - sl*_dl*fac1 + sr*_dr*fac2)/fac3

                for k in range(dim):
                    fmv[k][i] = ((_dl*vl[k][i]*Vnl + _pl*nx[k][i])*fac1 - (_dr*vr[k][i]*Vnr + _pr*nx[k][i])*fac2 \
                            - sl*(_dl*vl[k][i])*fac1 + sr*(_dr*vr[k][i])*fac2)/fac3

                el = 0.5*_dl*vl_sq + _pl/(gamma - 1.0)
                er = 0.5*_dr*vr_sq + _pr/(gamma - 1.0)
                fe.data[i]  = ((el + _pl)*Vnl*fac1 - (er + _pr)*Vnr*fac2 - sl*el*fac1 + sr*er*fac2)/fac3

            else:

                # right state
                fm.data[i]  = _dr*(Vnr - wn)
                fe.data[i]  = (0.5*_dr*vr_sq + _pr/(gamma - 1.0))*(Vnr - wn) + _pr*Vnr

                for k in range(dim):
                    fmv[k][i] = _dr*vr[k][i]*(Vnr - wn) + _pr*nx[k][i]

    cdef void get_waves(self, double d_l, double u_l, double p_l,
            double d_r, double u_r, double p_r,
            double gamma, double *sl, double *sc, double *sr):

        cdef double p_star, u_star
        cdef double d_avg, c_avg

        cdef double s_l, s_r, s_c
        cdef double c_l, c_r

        cdef double z, p_lr
        cdef double Q = 2.0
        cdef double A_l, A_r, B_l, B_r
        cdef double g_l, g_r

        cdef double p_min
        cdef double p_max
        cdef double c_floor = 1.0E-10

        c_l = fmax(sqrt(gamma*p_l/d_l), c_floor)
        c_r = fmax(sqrt(gamma*p_r/d_r), c_floor)

        d_avg = 0.5*(d_l + d_r)
        c_avg = 0.5*(c_l + c_r)

        # estimate p* - eq. 9.20
        p_star = fmax(0.0, 0.5*(p_l + p_r) + 0.5*(u_l - u_r)*d_avg*c_avg)

        p_min = fmin(p_l, p_r)
        p_max = fmax(p_l, p_r)

        if(((p_max/p_min) < Q) and ((p_min < p_star) and (p_star < p_max))):

            u_star = 0.5*(u_l + u_r) + 0.5*(p_l - p_r)/(d_avg*c_avg)

        elif(p_star <= p_min):

            # two rarefaction riemann solver (TRRS)
            # eq. 9.31
            z = (gamma - 1.0)/(2.0*gamma);

            # eq. 9.35
            p_lr = pow(p_l/p_r, z);

            u_star = (p_lr*u_l/c_l + u_r/c_r + 2.0*(p_lr - 1.0)/(gamma - 1.0))/\
                    (p_lr/c_l + 1.0/c_r)

            # estimate p* from two rarefaction aprroximation - eq. 9.36
            p_star  = 0.5*p_l*pow(1.0 + (gamma - 1.0)*(u_l - u_star)/(2.0*c_l), 1.0/z)
            p_star += 0.5*p_r*pow(1.0 + (gamma - 1.0)*(u_star - u_r)/(2.0*c_r), 1.0/z)


        else:

            # two shock riemann solver (TSRS)
            # eq. 9.31
            A_l = 2.0/((gamma + 1.0)*d_l)
            A_r = 2.0/((gamma + 1.0)*d_r)

            B_l = p_l*((gamma - 1.0)/(gamma + 1.0))
            B_r = p_r*((gamma - 1.0)/(gamma + 1.0))

            # 9.41
            g_l = sqrt(A_l/(p_star + B_l))
            g_r = sqrt(A_r/(p_star + B_r))

            # estimate p* from two shock aprroximation - eq. 9.43
            p_star = (g_l*p_l + g_r*p_r - (u_r- u_l))/(g_l + g_r)
            u_star = 0.5*(u_l + u_r) + 0.5*(g_r*(p_star - p_r) - g_l*(p_star - p_l))


        # calculate fastest left wave speed estimates - eq. 10.68-10.69
        if(p_star <= p_l):
            # rarefaction wave
            s_l = u_l - c_l

        else:
            # shock wave
            s_l = u_l - c_l*sqrt(1.0+((gamma+1.0)/(2.0*gamma))*(p_star/p_l - 1.0))

        # calculate fastest right wave speed estimates - eq. 10.68-10.69
        if(p_star <= p_r):
            # Rarefaction wave
            s_r = u_r+ c_r

        else:
            # shock wave
            s_r = u_r+ c_r*sqrt(1.0+((gamma+1.0)/(2.0*gamma))*(p_star/p_r - 1.0))


        # contact wave speed - eq. 10.70
        s_c = (p_r - p_l + d_l*u_l*(s_l - u_l) - d_r*u_r*(s_r- u_r))/(d_l*(s_l- u_l) - d_r*(s_r- u_r))

        sl[0] = s_l
        sc[0] = s_c
        sr[0] = s_r

cdef class HLLC(HLL):

    cdef _solve(self, CarrayContainer fluxes, CarrayContainer left_faces, CarrayContainer right_faces, CarrayContainer faces,
            double t, double dt, int iteration_count, int dim):

        # left state primitive variables
        cdef DoubleArray dl = left_faces.get_carray("density")
        cdef DoubleArray pl = left_faces.get_carray("pressure")

        # left state primitive variables
        cdef DoubleArray dr = right_faces.get_carray("density")
        cdef DoubleArray pr = right_faces.get_carray("pressure")

        cdef DoubleArray fm = fluxes.get_carray("mass")
        cdef DoubleArray fe = fluxes.get_carray("energy")

        # local variables
        cdef int i, k
        cdef double factor_1, factor_2, frho
        cdef double _dl, _pl, el
        cdef double _dr, _pr, er
        cdef double wn, Vnl, Vnr, sl, sr, s_contact
        cdef double vl_tmp, vr_tmp, nx_tmp, vl_sq, vr_sq
        cdef np.float64_t *vl[3], *vr[3], *fmv[3], *nx[3], *wx[3]

        cdef double fac1, fac2, fac3

        cdef double gamma = self.gamma
        cdef int num_faces = faces.get_number_of_items()

        left_faces.pointer_groups(vl,  left_faces.named_groups['velocity'])
        right_faces.pointer_groups(vr, right_faces.named_groups['velocity'])
        fluxes.pointer_groups(fmv, fluxes.named_groups['momentum'])
        faces.pointer_groups(nx, faces.named_groups['normal'])
        faces.pointer_groups(wx, faces.named_groups['velocity'])

        for i in range(num_faces):

            # left state
            _dl = dl.data[i]
            _pl = pl.data[i]

            # right state
            _dr = dr.data[i]
            _pr = pr.data[i]

            Vnl = Vnr = 0.0
            vl_sq = vr_sq = wn = 0.0
            for k in range(dim):

                vl_tmp = vl[k][i]; vr_tmp = vr[k][i]
                nx_tmp = nx[k][i]

                # left/right velocity square
                vl_sq += vl_tmp*vl_tmp
                vr_sq += vr_tmp*vr_tmp

                # project left/righ velocity to face normal
                Vnl += vl_tmp*nx_tmp
                Vnr += vr_tmp*nx_tmp

                # project face velocity to face normal
                wn += wx[k][i]*nx_tmp

            self.get_waves(_dl, Vnl, _pl, _dr, Vnr, _pr, gamma,
                    &sl, &s_contact, &sr)

            # calculate interface flux - eq. 10.71
            if(wn <= sl):

                # left state
                fm.data[i]  = _dl*(Vnl - wn)
                fe.data[i]  = (0.5*_dl*vl_sq + _pl/(gamma - 1.0))*(Vnl - wn) + _pl*Vnl

                for k in range(dim):
                    fmv[k][i] = _dl*vl[k][i]*(Vnl - wn) + _pl*nx[k][i]

            elif((sl < wn) and (wn <= sr)):

                # intermediate state
                if(wn <= s_contact):

                    # left star state - eq. 10.38 and 10.39
                    factor_1 = _dl*(sl - Vnl)/(sl - s_contact)
                    factor_2 = factor_1*(sl - wn)*(s_contact - Vnl) + _pl
                    frho = _dl*(Vnl - sl) + factor_1*(sl - wn)

                    # total energy
                    el = 0.5*_dl*vl_sq + _pl/(gamma-1.0)

                    fm.data[i] = frho
                    fe.data[i] = (el + _pl)*Vnl - sl*el +\
                            (sl - wn)*factor_1*(el/_dl + (s_contact - Vnl)*\
                            (s_contact + _pl/(_dl*(sl - Vnl))))

                    for k in range(dim):
                        fmv[k][i] = frho*vl[k][i] + factor_2*nx[k][i]

                else:

                    # right star state
                    factor_1 = _dr*(sr - Vnr)/(sr - s_contact)
                    factor_2 = factor_1*(sr - wn)*(s_contact - Vnr) + _pr
                    frho = _dr*(Vnr - sr) + factor_1*(sr - wn)

                    # total energy
                    er = 0.5*_dr*vr_sq + _pr/(gamma-1.0)

                    fm.data[i] = frho
                    fe.data[i] = (er + _pr)*Vnr - sr*er +\
                            (sr - wn)*factor_1*(er/_dr + (s_contact - Vnr)*\
                            (s_contact + _pr/(_dr*(sr - Vnr))))

                    for k in range(dim):
                        fmv[k][i] = frho*vr[k][i] + factor_2*nx[k][i]

            else:

                # right state
                fm.data[i]  = _dr*(Vnr - wn)
                fe.data[i]  = (0.5*_dr*vr_sq + _pr/(gamma - 1.0))*(Vnr - wn) + _pr*Vnr

                for k in range(dim):
                    fmv[k][i] = _dr*vr[k][i]*(Vnr - wn) + _pr*nx[k][i]
