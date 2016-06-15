from ..containers.containers cimport CarrayContainer
from ..utils.carray cimport DoubleArray
from ..reconstruction.reconstruction cimport ReconstructionBase

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, pow, fmin, fmax

cdef class RiemannBase:
    def __init__(self, ReconstructionBase reconstruction, double gamma=1.4, double cfl=0.3):
        self.reconstruction = reconstruction
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
        cdef double vl_tmp, vr_tmp, nx_tmp, vl_mag, vr_mag
        cdef np.float64_t *vl[3], *vr[3], *fmv[3], *nx[3], *wx[3]

        cdef double gamma = self.gamma
        cdef int num_faces = faces.get_number_of_items()


        left_faces.extract_field_vec_ptr(vl, "velocity")
        right_faces.extract_field_vec_ptr(vr, "velocity")
        fluxes.extract_field_vec_ptr(fmv, "momentum")
        faces.extract_field_vec_ptr(nx, "normal")
        faces.extract_field_vec_ptr(wx, "velocity")

        for i in range(num_faces):

            # left state
            _dl = dl.data[i]
            _pl = pl.data[i]

            # right state
            _dr = dr.data[i]
            _pr = pr.data[i]

            Vnl = Vnr = 0.0
            vl_mag = vr_mag = wn = 0.0
            for k in range(dim):

                vl_tmp = vl[k][i]; vr_tmp = vr[k][i]
                nx_tmp = nx[k][i]

                # left/right velocity magnitude
                vl_mag += vl_tmp*vl_tmp
                vr_mag += vr_tmp*vr_tmp

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
                fe.data[i]  = (0.5*_dl*vl_mag + _pl/(gamma - 1.0))*(Vnl - wn) + _pl*Vnl

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

                el = 0.5*_dl*vl_mag + _pl/(gamma - 1.0)
                er = 0.5*_dr*vr_mag + _pr/(gamma - 1.0)
                fe.data[i]  = ((el + _pl)*Vnl*fac1 - (er + _pr)*Vnr*fac2 - sl*el*fac1 + sr*er*fac2)/fac3

            else:

                # right state
                fm.data[i]  = _dr*(Vnr - wn)
                fe.data[i]  = (0.5*_dr*vr_mag + _pr/(gamma - 1.0))*(Vnr - wn) + _pr*Vnr

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

#cdef class HLLC(HLL):
#
#    cdef _solve(self, CarrayContainer fluxes, CarrayContainer left_faces, CarrayContainer right_faces, CarrayContainer faces,
#            double t, double dt, int iteration_count):
#
#        # left state primitive variables
#        cdef DoubleArray dl = left_faces.get_carray("density")
#        cdef DoubleArray ul = left_faces.get_carray("velocity-x")
#        cdef DoubleArray vl = left_faces.get_carray("velocity-y")
#        cdef DoubleArray pl = left_faces.get_carray("pressure")
#
#        # left state primitive variables
#        cdef DoubleArray dr = right_faces.get_carray("density")
#        cdef DoubleArray ur = right_faces.get_carray("velocity-x")
#        cdef DoubleArray vr = right_faces.get_carray("velocity-y")
#        cdef DoubleArray pr = right_faces.get_carray("pressure")
#
#        cdef DoubleArray fm  = fluxes.get_carray("mass")
#        cdef DoubleArray fmu = fluxes.get_carray("momentum-x")
#        cdef DoubleArray fmv = fluxes.get_carray("momentum-y")
#        cdef DoubleArray fe  = fluxes.get_carray("energy")
#
#        cdef DoubleArray nx = faces.get_carray("normal-x")
#        cdef DoubleArray ny = faces.get_carray("normal-y")
#        cdef DoubleArray wx = faces.get_carray("velocity-x")
#        cdef DoubleArray wy = faces.get_carray("velocity-y")
#
#        # local variables
#        cdef int i
#        cdef double _nx, _ny
#        cdef double factor_1, factor_2
#        cdef double _dl, _ul, _vl, _pl, _el
#        cdef double _dr, _ur, _vr, _pr, _er
#        cdef double _wn, _Vnl, _Vnr, _sl, _sr, s_contact
#
#        cdef double gamma = self.gamma
#        cdef int num_faces = faces.get_number_of_items()
#
#        for i in range(num_faces):
#
#            # left state
#            _dl = dl.data[i]
#            _ul = ul.data[i]
#            _vl = vl.data[i]
#            _pl = pl.data[i]
#
#            # right state
#            _dr = dr.data[i]
#            _ur = ur.data[i]
#            _vr = vr.data[i]
#            _pr = pr.data[i]
#
#            # face normal
#            _nx = nx.data[i]
#            _ny = ny.data[i]
#
#            # project velocity onto face normal
#            _Vnl = _ul*_nx + _vl*_ny
#            _Vnr = _ur*_nx + _vr*_ny
#
#            self.get_waves(_dl, _Vnl, _pl, _dr, _Vnr, _pr, gamma,
#                    &_sl, &s_contact, &_sr)
#
#            # velocity of face projected onto face normal
#            _wn = wx.data[i]*_nx + wy.data[i]*_ny
#
#            # calculate interface flux - eq. 10.71
#            if(_wn <= _sl):
#
#                # left state
#                fm.data[i]  = _dl*(_Vnl - _wn)
#                fmu.data[i] = _dl*_ul*(_Vnl - _wn) + _pl*_nx
#                fmv.data[i] = _dl*_vl*(_Vnl - _wn) + _pl*_ny
#                fe.data[i]  = (0.5*_dl*(_ul*_ul + _vl*_vl) + _pl/(gamma - 1.0))*(_Vnl - _wn) + _pl*_Vnl
#
#            elif((_sl < _wn) and (_wn <= _sr)):
#
#                # intermediate state
#                if(_wn <= s_contact):
#
#                    # left star state
#                    factor_1 = _dl*(_sl - _Vnl)/(_sl - s_contact)
#                    factor_2 = factor_1*(_sl - _wn)*(s_contact - _Vnl) + _pl
#
#                    # density flux
#                    frho = _dl*(_Vnl - _sl) + factor_1*(_sl - _wn)
#
#                    fm.data[i]  = frho
#                    fmu.data[i] = frho*_ul + factor_2*_nx
#                    fmv.data[i] = frho*_vl + factor_2*_ny
#
#                    # total energy
#                    _el = 0.5*_dl*(_ul*_ul + _vl*_vl) + _pl/(gamma-1.0)
#
#                    fe.data[i] = (_el + _pl)*_Vnl - _el*_sl +\
#                            (_sl - _wn)*factor_1*(_el/_pl + (s_contact - _Vnl)*\
#                            (s_contact + _pl/(_dl*(_sl - _Vnl))))
#
#                else:
#
#                    # right star state
#                    factor_1 = _dr*(_sr - _Vnr)/(_sr - s_contact)
#                    factor_2 = factor_1*(_sr - _wn) *(s_contact - _Vnr) + _pr
#
#                    # density flux
#                    frho = _dr*(_Vnr - _sr) + factor_1*(_sr - _wn)
#
#                    fm.data[i]  = frho
#                    fmu.data[i] = frho*_ur + factor_2*_nx
#                    fmv.data[i] = frho*_vr + factor_2*_ny
#
#                    # total energy
#                    _er = 0.5*_dr*(_ur*_ur + _vr*_vr) + _pr*gamma/(gamma-1.0)
#
#                    fe.data[i] = (_er + _pr)*_Vnr - _er*_sr +\
#                            (_sr - _wn)*factor_1*(_er/_pr + (s_contact - _Vnr)*\
#                            (s_contact + _pr/(_dr*(_sr - _Vnr))))
#
#            else:
#
#                # right state
#                fm.data[i]  = _dr*(_Vnr - _wn)
#                fmu.data[i] = _dr*_ur*(_Vnr - _wn) + _pr*_nx
#                fmv.data[i] = _dr*_vr*(_Vnr - _wn) + _pr*_ny
#                fe.data[i]  = (0.5*_dr*(_ur*_ur + _vr*_vr) + _pr/(gamma - 1.0))*(_Vnr - _wn) + _pr*_Vnr
