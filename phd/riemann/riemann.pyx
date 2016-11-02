from ..containers.containers cimport CarrayContainer
from ..utils.carray cimport DoubleArray
from ..reconstruction.reconstruction cimport ReconstructionBase

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, pow, fmin, fmax, fabs

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

    cdef _deboost(self, CarrayContainer fluxes, CarrayContainer faces, int dim):

        cdef DoubleArray fm = fluxes.get_carray("mass")
        cdef DoubleArray fe = fluxes.get_carray("energy")

        cdef np.float64_t *fmv[3], *wx[3]

        cdef int m, k
        cdef int num_faces = faces.get_number_of_items()

        fluxes.pointer_groups(fmv, fluxes.named_groups['momentum'])
        faces.pointer_groups(wx, faces.named_groups['velocity'])

        # return flux to lab frame Pakmor 2011
        for m in range(num_faces):
            for k in range(dim):
                fe.data[m] += wx[k][m]*(0.5*wx[k][m]*fm.data[m] + fmv[k][m])
                fmv[k][m]  += wx[k][m]*fm.data[m]


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
        cdef int boost = self.reconstruction.boost
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

            # if boosted we are in face frame
            if boost == 1:
                wn = 0.

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

        if boost == 1:
            self._deboost(fluxes, faces, dim)

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
        cdef int boost = self.reconstruction.boost
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

            # if boosted we are in face frame
            if boost == 1:
                wn = 0.

            # hack - delete later >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if _dl < 0. or _dr < 0.:
                raise RuntimeError("Vacuum L/R state")
            _cl = sqrt(gamma*_pl/_dl); _cr = sqrt(gamma*_pr/_dr)
            if (2.*_cl/(gamma-1.) + 2.*_cr/(gamma-1.)) <= (Vnr - Vnl):
                raise RuntimeError("Vacuum creation state")
            # hack - delete later <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 

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

        if boost == 1:
            self._deboost(fluxes, faces, dim)

cdef class Exact(RiemannBase):

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

        cdef int i, k
        cdef np.float64_t *vl[3], *vr[3], *fmv[3], *nx[3]

        # state values
        cdef double _dl, _pl
        cdef double _dr, _pr
        cdef double _d, v[3], _p

        cdef double vnl, vnr, vl_sq, vr_sq

        # wave estimates
        cdef double fr, fl, u_tmp
        cdef double p_star, u_star
        cdef double s_hl, s_tl, sl, sr
        cdef double c, cl, cr, c_star_l, c_star_r

        cdef double gamma = self.gamma
        cdef int num_faces = faces.get_number_of_items()

        left_faces.pointer_groups(vl,  left_faces.named_groups['velocity'])
        right_faces.pointer_groups(vr, right_faces.named_groups['velocity'])
        fluxes.pointer_groups(fmv, fluxes.named_groups['momentum'])
        faces.pointer_groups(nx, faces.named_groups['normal'])

        for i in range(num_faces):

            # left state
            _dl = dl.data[i]
            _pl = pl.data[i]

            # right state
            _dr = dr.data[i]
            _pr = pr.data[i]

            # sound speed
            cl = sqrt(gamma*_pl/_dl)
            cr = sqrt(gamma*_pr/_dr)

            vnl = vnr = 0.0
            vl_sq = vr_sq = 0.0
            for k in range(dim):

                # project left/righ velocity to face normal
                vnl += vl[k][i]*nx[k][i]
                vnr += vr[k][i]*nx[k][i]

                # left/right velocity square
                vl_sq += vl[k][i]*vl[k][i]
                vr_sq += vr[k][i]*vr[k][i]

            # newton rhapson 
            p_star = self.get_pstar(_dl, vnl, _pl, _dr, vnr, _pr, gamma)

            # calculate the contact wave speed
            fl = self.p_func(_dl, vnl, _pl, gamma, p_star)
            fr = self.p_func(_dr, vnr, _pr, gamma, p_star)
            u_star = 0.5*(vnl + vnr + fr - fl)

            if(0.0 <= u_star):

                # left of the contact discontinuity
                if(p_star <= _pl):

                    # left rarefraction, sound speed of head
                    s_hl = vnl - cl

                    if(0.0 <= s_hl):

                        # left state
                        fm.data[i]  = _dl*vnl
                        fe.data[i]  = (0.5*_dl*vl_sq + gamma*_pl/(gamma - 1.0))*vnl

                        for k in range(dim):
                            fmv[k][i] = _dl*vl[k][i]*vnl + _pl*nx[k][i]

                    else:

                        # sound speed of star state and tail of rarefraction
                        c_star_l = cl*pow(p_star/_pl, (gamma - 1.0)/(2.0*gamma))
                        s_tl = u_star - c_star_l

                        if(0.0 >= s_tl):

                            # star left state
                            _d = _dl*pow(p_star/_pl, 1.0/gamma)
                            _p = p_star

                            v_sq = vn = 0.
                            for k in range(dim):
                                v[k]  = vl[k][i] + (u_star - vnl)*nx[k][i]
                                vn   += v[k]*nx[k][i]
                                v_sq += v[k]*v[k]

                            fm.data[i]  = _d*vn
                            fe.data[i]  = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn

                            for k in range(dim):
                                fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]

                        else:

                            # inside left fan
                            c  = (2.0/(gamma + 1.0))*(cl + 0.5*(gamma - 1.0)*vnl)
                            _d = _dl*pow(c/cl, 2.0/(gamma - 1.0))
                            _p = _pl*pow(c/cl, 2.0*gamma/(gamma - 1.0))

                            v_sq = vn = 0.
                            for k in range(dim):
                                v[k]  = vl[k][i] + (c - vnl)*nx[k][i]
                                vn   += v[k]*nx[k][i]
                                v_sq += v[k]*v[k]

                            fm.data[i] = _d*vn
                            fe.data[i] = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn

                            for k in range(dim):
                                fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]
                else:

                    # left shock
                    sl = vnl - cl*sqrt((gamma + 1.0)*p_star/(2.0*gamma*_pl) + (gamma - 1.0)/(2.0*gamma))

                    if(0.0 <= sl):

                        # left state
                        fm.data[i] = _dl*vnl
                        fe.data[i] = (0.5*_dl*vl_sq + gamma*_pl/(gamma - 1.0))*vnl

                        for k in range(dim):
                            fmv[k][i] = _dl*vl[k][i]*vnl + _pl*nx[k][i]

                    else:

                        # star left state
                        _d = _dl*(p_star/_pl + (gamma - 1.0)/(gamma + 1.0))/(p_star*(gamma - 1.0)/((gamma + 1.0)*_pl) + 1.0)
                        _p = p_star

                        v_sq = vn = 0.
                        for k in range(dim):
                            v[k]  = vl[k][i] + (u_star - vnl)*nx[k][i]
                            vn   += v[k]*nx[k][i]
                            v_sq += v[k]*v[k]

                        fm.data[i]  = _d*vn
                        fe.data[i]  = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn

                        for k in range(dim):
                            fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]

            else:

                # right of the contact
                if(p_star >= _pr):

                    # right shock
                    sr = vnr + cr*sqrt((gamma + 1.0)*p_star/(2.0*gamma*_pr) + (gamma-1.0)/(2.0*gamma))

                    if(0.0 >= sr):

                        # right data states
                        fm.data[i] = _dr*vnr
                        fe.data[i] = (0.5*_dr*vr_sq + gamma*_pr/(gamma - 1.0))*vnr

                        for k in range(dim):
                            fmv[k][i] = _dr*vr[k][i]*vnr + _pr*nx[k][i]

                    else:

                        # star right state
                        _d = _dr*(p_star/_pr + (gamma - 1.0)/(gamma + 1.0))/(p_star*(gamma - 1.0)/((gamma + 1.0)*_pr) + 1.0)
                        _p = p_star

                        v_sq = vn = 0.
                        for k in range(dim):
                            v[k]  = vr[k][i] + (u_star - vnr)*nx[k][i]
                            vn   += v[k]*nx[k][i]
                            v_sq += v[k]*v[k]

                        fm.data[i]  = _d*vn
                        fe.data[i]  = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn

                        for k in range(dim):
                            fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]

                else:

                    # right rarefaction
                    s_hr = vnr + cr

                    if(0.0 >= s_hr):

                        # right data state
                        fm.data[i]  = _dr*vnr
                        fe.data[i]  = (0.5*_dr*vr_sq + gamma*_pr/(gamma - 1.0))*vnr

                        for k in range(dim):
                            fmv[k][i] = _dr*vr[k][i]*vnr + _pr*nx[k][i]

                    else:

                        # sound speed of the star state and sound speed
                        # of the tail of the rarefraction
                        c_star_r = cr*pow(p_star/_pr, (gamma-1.0)/(2.0*gamma))
                        s_tr = u_star + c_star_r

                        if(0.0 <= s_tr):

                            # star left state
                            _d = _dr*pow(p_star/_pr, 1.0/gamma)
                            _p = p_star

                            v_sq = vn = 0.
                            for k in range(dim):
                                v[k]  = vr[k][i] + (u_star - vnr)*nx[k][i]
                                vn   += v[k]*nx[k][i]
                                v_sq += v[k]*v[k]

                            fm.data[i]  = _d*vn
                            fe.data[i]  = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn

                            for k in range(dim):
                                fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]

                        else:

                            # sampled point is inside right fan
                            c = (2.0/(gamma + 1.0))*(cr - 0.5*(gamma - 1.0)*vnr)
                            u_tmp = (2.0/(gamma + 1.0))*(-cr + 0.5*(gamma-1.0)*vnr)

                            _d = _dr*pow(c/cr, 2.0/(gamma - 1.0))
                            _p = _pr*pow(c/cr, 2.0*gamma/(gamma - 1.0))

                            v_sq = vn = 0.
                            for k in range(dim):
                                v[k]  = vr[k][i] + (u_tmp - vnr)*nx[k][i]
                                vn   += v[k]*nx[k][i]
                                v_sq += v[k]*v[k]

                            fm.data[i]  = _d*vn
                            fe.data[i]  = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn

                            for k in range(dim):
                                fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]

        self._deboost(fluxes, faces, dim)

    cdef p_guess(self, double d_l, double u_l, double p_l, double d_r, double u_r, double p_r, double gamma):

        cdef double c_l, c_r
        cdef double ppv
        cdef double gl, gr, p_0
        cdef double p_star, p_max, p_min, q_max
        cdef double p_lr, p_tl

        c_l = sqrt(gamma*p_l/d_l)
        c_r = sqrt(gamma*p_r/d_r)

        # initial guess for pressure
        ppv = 0.5*(p_l + p_r) - 0.125*(u_r - u_l)*(d_l + d_r)*(c_l + c_r)

        p_star = max(0.0, ppv)
        p_max  = max(p_l, p_r)
        p_min  = min(p_l, p_r)
        q_max  = p_max/p_min

        if ((q_max <= 2.0) and (p_min <= ppv <= p_max)):

            p_0 = ppv

        elif (ppv <= p_min):

            p_lr   = pow(p_l/p_r, (gamma - 1.0)/(2.0*gamma))
            u_star = (p_lr*u_l/c_l + u_r/c_r + 2*(p_lr - 1)/(gamma - 1))
            u_star = u_star/(p_lr/c_l + 1/c_r)
            p_tl   = pow(1 + (gamma - 1)*(u_l - u_star)/(2*c_l), 2.0*gamma/(gamma - 1.0))
            p_tr   = pow(1 + (gamma - 1)*(u_star - u_r)/(2*c_r), 2.0*gamma/(gamma - 1.0))

            p_0 = 0.5*(p_l*p_tl + p_r*p_tr)

        else:

            gl = sqrt((2.0/(d_l*(gamma+1)))/((gamma-1)*p_l/(gamma+1) + ppv))
            gr = sqrt((2.0/(d_r*(gamma+1)))/((gamma-1)*p_r/(gamma+1) + ppv))

            p_0 = (gl*p_l + gr*p_r - (u_r - u_l))/(gr + gl)


        return p_0

    cdef p_func(self, double d, double u, double p, double gamma, double p_old):

        cdef double f
        cdef double c = sqrt(gamma*p/d)
        cdef double Ak, Bk

        # rarefaction wave
        if (p_old <= p):
            f = 2*c/(gamma-1)*(pow(p_old/p, (gamma-1)/(2*gamma)) - 1)

        # shock wave
        else:
            Ak = 2/(d*(gamma+1))
            Bk = p*(gamma-1)/(gamma+1)

            f = (p_old - p)*sqrt(Ak/(p_old + Bk))

        return f


    cdef p_func_deriv(self, double d, double u, double p, double gamma, double p_old):

        cdef double df
        cdef double c = sqrt(gamma*p/d)
        cdef double Ak, Bk

        # rarefaction wave
        if (p_old <= p):
            df = 1/(c*d)*pow(p_old/p, -(gamma + 1)/(2*gamma))

        # shock wave
        else:
            Ak = 2/(d*(gamma + 1))
            Bk = p*(gamma - 1)/(gamma + 1)

            df = sqrt(Ak/(p_old + Bk))*(1.0 - 0.5*(p_old - p)/(Bk + p_old))

        return df


    cdef get_pstar(self, double d_l, double u_l, double p_l, double d_r, double u_r, double p_r, double gamma):

        cdef double p_old
        cdef double u_diff = u_r - u_l
        cdef int i = 0
        cdef double TOL = 1.0E-6
        cdef double change, p, f_r, f_l, df_r, df_l
        cdef int MAX_ITER = 1000

        p_old = self.p_guess(d_l, u_l, p_l, d_r, u_r, p_r, gamma)

        while (i < MAX_ITER):

            f_l  = self.p_func(d_l, u_l, p_l, gamma, p_old)
            f_r  = self.p_func(d_r, u_r, p_r, gamma, p_old)
            df_l = self.p_func_deriv(d_l, u_l, p_l, gamma, p_old)
            df_r = self.p_func_deriv(d_r, u_r, p_r, gamma, p_old)

            p = p_old - (f_l + f_r + u_diff)/(df_l + df_r)

            change = 2.0*fabs((p - p_old)/(p + p_old))
            if (change <= TOL):
                return p

            if (p < 0.0):
                p = TOL

            p_old = p
            i += 1


        # exit failure due to divergence
        print "did not converge"
        return p
