import numpy as np
from collections import defaultdict

cimport cython
cimport numpy as np
from libc.math cimport sqrt, pow, fmin, fmax, fabs

from ..utils.particle_tags import ParticleTAGS
from ..utils.carray cimport DoubleArray, IntArray

cdef int Real = ParticleTAGS.Real

cdef class RiemannBase:
    """
    Riemann base that all riemann solvers need to inherit.
    """
    def __init__(self, double param_cfl=0.5, bint param_boost=True):
        self.param_cfl = param_cfl
        self.param_boost = param_boost
        self.registered_fields = False

    def initialize(self):
        """
        Setup all connections for computation classes. Should check always
        check if registered_fields is True.
        """
        msg = "Reconstruction::initialize called!"
        raise NotImplementedError(msg)

    def set_fields_for_flux(CarrayContainer particles):
        """
        Create lists of variables to calculate fluxes.
        """
        cdef str field, dtype
        cdef dict field_types = {}, named_groups = defaultdict([])

        # add standard primitive fields
        for field in particles.named_groups["conserative"]:

            # add field to group
            named_groups["conserative"].append(field)

            # check type of field
            dtype = particles.carray_info[field]
            if dtype != "double":
                raise RuntimeError(
                        "Reconstruction: %field non double type" %\
                         dtype)
            field_types[field] = "double"

        named_groups['momentum'] = particles.named_group['momentum']

        # store fields info
        self.registered_fields = True
        self.flux_fields = field_types
        self.flux_field_groups = named_groups

    cpdef compute_fluxes(self, CarrayContainer particles, Mesh mesh, ReconstructionBase reconstruction,
            EquationStateBase eos, int dim):
        """Compute fluxes for each face in the mesh"""

        self.fluxes.resize(mesh.faces.get_number_of_items())
        self.riemann_solver(particles, mesh, reconstruction, eos.get_gamma(), dim)

    cdef riemann_solver(CarrayContainer particles, Mesh mesh, ReconstructionBase reconstruction,
            double gamma, int dim):
        """Riemann solver"""
        msg = "RiemannBase::riemann_solver called!"
        raise NotImplementedError(msg)

    cpdef double compute_time_step(self, CarrayContainer particles, EquationStateBase eos, int dim):
        """
        Compute time step for next integration step.
        """
        cdef IntArray tags  = particles.get_carray("tag")
        cdef DoubleArray vol = particles.get_carray("volume")

        cdef DoubleArray d = particles.get_carray("density")
        cdef DoubleArray p = particles.get_carray("pressure")

        cdef int i, k
        cdef np.float64_t* v[3]
        cdef double c, R, dt, vsq
        cdef bint boost = self.param_boost

        particles.pointer_groups(v, particles.named_groups['velocity'])
        c = eos.sound_speed(d.data[i], p.data[i])

        if dim == 2:
            R = sqrt(vol.data[0]/np.pi)
        elif dim == 3:
            R = pow(3.0*vol.data[0]/(4.0*np.pi), 1.0/3.0)

        vsq = 0.0
        for k in range(dim):
            vsq += v[k][0]*v[k][0]

        if boost:
            dt = R/c
        else:
            dt = R/(c + sqrt(vsq))

        for i in range(particles.get_number_of_items()):
            if tags.data[i] == Real:
                c = eos.sound_speed(d.data[i], p.data[i])

                # calculate approx radius of each voronoi cell
                if dim == 2:
                    R = sqrt(vol.data[i]/np.pi)
                elif dim == 3:
                    R = pow(3.0*vol.data[i]/(4.0*np.pi), 1.0/3.0)

                vsq = 0.0
                for k in range(dim):
                    vsq += v[k][i]*v[k][i]

                if boost:
                    dt = fmin(R/c, dt)
                else:
                    dt = fmin(R/(c + sqrt(vsq)), dt)
        return dt

    cdef deboost(self, CarrayContainer fluxes, CarrayContainer faces, int dim):
        """
        Deboost riemann solution of fluxes from face reference to lab frame.
        """
        cdef DoubleArray fm = fluxes.get_carray("mass")
        cdef DoubleArray fe = fluxes.get_carray("energy")

        cdef int m, k
        cdef np.float64_t *fmv[3], *wx[3]

        fluxes.pointer_groups(fmv, fluxes.named_groups['momentum'])
        faces.pointer_groups(wx, faces.named_groups['velocity'])

        # return flux to lab frame (Pakmor 2011)
        for m in range(faces.get_number_of_items()):
            for k in range(dim):
                fe.data[m] += wx[k][m]*(0.5*wx[k][m]*fm.data[m] + fmv[k][m])
                fmv[k][m]  += wx[k][m]*fm.data[m]

cdef class HLL(RiemannBase):
    def __init__(self, double param_cfl=0.5, bint param_boost=True):
        super(HLL, self).__init__(param_cfl, param_boost)

    def initialize(self):
        if not self.registered_fields:
            raise RuntimeError(
                    "Riemann did not set fields for flux!")

        self.fluxes = CarrayContainer(var_dict=self.flux_fields)
        self.fluxes.named_groups = self.flux_field_groups

    cdef riemann_solver(CarrayContainer particles, Mesh mesh, ReconstructionBase reconstruction,
            double gamma, int dim):

        # left state primitive variables
        cdef DoubleArray dl = reconstruction.left_state.get_carray("density")
        cdef DoubleArray pl = reconstruction.left_state.get_carray("pressure")

        # right state primitive variables
        cdef DoubleArray dr = reconstruction.right_state.get_carray("density")
        cdef DoubleArray pr = recnostruction.right_state.get_carray("pressure")

        # face mass and energy flux
        cdef DoubleArray fm = self.fluxes.get_carray("mass")
        cdef DoubleArray fe = self.fluxes.get_carray("energy")

        cdef int i, k
        cdef double _dl, _pl
        cdef double _dr, _pr
        cdef double fac1, fac2, el, er
        cdef double wn, Vnl, Vnr, sl, sr, s_contact
        cdef double vl_tmp, vr_tmp, nx_tmp, vl_sq, vr_sq
        cdef np.float64_t *vl[3], *vr[3], *fmv[3], *nx[3], *wx[3]

        cdef bint boost = self.param_boost
        cdef int num_faces = mesh.faces.get_number_of_items()

        # particle velocities left/right face
        reconstruction.left_state.pointer_groups(vl,
                reconstruction.left_state.named_groups['velocity'])
        reconstruction.right_state.pointer_groups(vr,
                reconstruction.right_state.named_groups['velocity'])

        # face momentum fluxes
        self.fluxes.pointer_groups(fmv, self.fluxes.named_groups['momentum'])

        # face normal and velocity
        mesh.faces.pointer_groups(nx, mesh.faces.named_groups['normal'])
        mesh.faces.pointer_groups(wx, mesh.faces.named_groups['velocity'])

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

            if boost:
                wn = 0. # in face frame

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

        if boost:
            self.deboost(self.fluxes, mesh.faces, dim)

    cdef inline void get_waves(self, double dl, double ul, double pl,
            double dr, double ur, double pr,
            double gamma, double *sl, double *sc, double *sr):

        cdef double p_star, u_star
        cdef double d_avg, c_avg

        cdef double _sl, _sr, _sc
        cdef double cl, cr

        cdef double z, plr
        cdef double Q = 2.
        cdef double Al, Ar, Bl, Br
        cdef double gl, gr

        cdef double p_min
        cdef double p_max
        cdef double c_floor = 1.0E-10

        cl = fmax(sqrt(gamma*pl/dl), c_floor)
        cr = fmax(sqrt(gamma*pr/dr), c_floor)

        d_avg = .5*(dl + dr)
        c_avg = .5*(cl + cr)

        # estimate p* - eq. 9.20
        p_star = fmax(0., .5*(pl + pr) + .5*(ul - ur)*d_avg*c_avg)

        p_min = fmin(pl, pr)
        p_max = fmax(pl, pr)

        if(((p_max/p_min) < Q) and ((p_min < p_star) and (p_star < p_max))):

            u_star = .5*(ul + ur) + .5*(pl - pr)/(d_avg*c_avg)

        elif(p_star <= p_min):

            # two rarefaction riemann solver (TRRS)
            # eq. 9.31
            z = (gamma - 1.)/(2.*gamma);

            # eq. 9.35
            plr = pow(pl/pr, z);

            u_star = (plr*ul/cl + ur/cr + 2.*(plr - 1.)/(gamma - 1.))/\
                    (plr/cl + 1./cr)

            # estimate p* from two rarefaction aprroximation - eq. 9.36
            p_star  = .5*pl*pow(1. + (gamma - 1.)*(ul - u_star)/(2.*cl), 1./z)
            p_star += .5*pr*pow(1. + (gamma - 1.)*(u_star - ur)/(2.*cr), 1./z)

        else:

            # two shock riemann solver (TSRS)
            # eq. 9.31
            Al = 2./((gamma + 1.)*dl)
            Ar = 2./((gamma + 1.)*dr)

            Bl = pl*((gamma - 1.)/(gamma + 1.))
            Br = pr*((gamma - 1.)/(gamma + 1.))

            # 9.41
            gl = sqrt(Al/(p_star + Bl))
            gr = sqrt(Ar/(p_star + Br))

            # estimate p* from two shock aprroximation - eq. 9.43
            p_star = (gl*pl + gr*pr - (ur - ul))/(gl + gr)
            u_star = .5*(ul + ur) + .5*(gr*(p_star - pr) - gl*(p_star - pl))

        # calculate fastest left wave speed estimates - eq. 10.68-10.69
        if(p_star <= pl):
            # rarefaction wave
            _sl = ul - cl

        else:
            # shock wave
            _sl = ul - cl*sqrt(1.+((gamma+1.)/(2.*gamma))*(p_star/pl - 1.))

        # calculate fastest right wave speed estimates - eq. 10.68-10.69
        if(p_star <= pr):
            # Rarefaction wave
            _sr = ur + cr

        else:
            # shock wave
            _sr = ur + cr*sqrt(1. + ((gamma+1.)/(2.*gamma))*(p_star/pr - 1.))

        # contact wave speed - eq. 10.70
        _sc = (pr - pl + dl*ul*(_sl - ul) - dr*ur*(_sr - ur))/(dl*(_sl - ul) - dr*(_sr - ur))

        sl[0] = _sl
        sc[0] = _sc
        sr[0] = _sr

#cdef class HLLC(HLL):
#
#    cdef riemann_solver(self, CarrayContainer fluxes, CarrayContainer left_faces, CarrayContainer right_faces,
#            CarrayContainer faces, double gamma, int dim):
#
#        # left state primitive variables
#        cdef DoubleArray dl = left_faces.get_carray("density")
#        cdef DoubleArray pl = left_faces.get_carray("pressure")
#
#        # left state primitive variables
#        cdef DoubleArray dr = right_faces.get_carray("density")
#        cdef DoubleArray pr = right_faces.get_carray("pressure")
#
#        cdef DoubleArray fm = fluxes.get_carray("mass")
#        cdef DoubleArray fe = fluxes.get_carray("energy")
#
#        # local variables
#        cdef int i, k
#
#        cdef double _dl, _pl, _vl[3], el, cl
#        cdef double _dr, _pr, _vr[3], er, cr
#        cdef double d, p, v[3], n[3], vn, vsq
#
#        cdef double factor_1, factor_2, frho
#        cdef double wn, Vnl, Vnr, sl, sr, s_contact
#        cdef double vl_tmp, vr_tmp, nx_tmp, vl_sq, vr_sq
#        cdef np.float64_t *vl[3], *vr[3], *fmv[3], *nx[3], *wx[3]
#
#        cdef double fac1, fac2, fac3
#
#        cdef int boost = self.param_boost
#        cdef int num_faces = faces.get_number_of_items()
#
#        left_faces.pointer_groups(vl,  left_faces.named_groups['velocity'])
#        right_faces.pointer_groups(vr, right_faces.named_groups['velocity'])
#        fluxes.pointer_groups(fmv, fluxes.named_groups['momentum'])
#        faces.pointer_groups(nx, faces.named_groups['normal'])
#        faces.pointer_groups(wx, faces.named_groups['velocity'])
#
#        for i in range(num_faces):
#
#            # left state
#            _dl = dl.data[i]
#            _pl = pl.data[i]
#
#            # right state
#            _dr = dr.data[i]
#            _pr = pr.data[i]
#
#            cl = sqrt(gamma*_pl/_dl)
#            cr = sqrt(gamma*_pr/_dr)
#
#            Vnl = Vnr = 0.0
#            vl_sq = vr_sq = wn = 0.0
#            for k in range(dim):
#
#                _vl[k] = vl[k][i]
#                _vr[k] = vr[k][i]
#
#                n[k] = nx[k][i]
#
#                # left/right velocity square
#                vl_sq += _vl[k]*_vl[k]
#                vr_sq += _vr[k]*_vr[k]
#
#                # project left/righ velocity to face normal
#                Vnl += _vl[k]*n[k]
#                Vnr += _vr[k]*n[k]
#
#                # project face velocity to face normal
#                wn += wx[k][i]*n[k]
#
#            # if boosted we are in face frame
#            if boost == 1:
#                wn = 0.
#
#            self.get_waves(_dl, Vnl, _pl, _dr, Vnr, _pr, gamma,
#                    &sl, &s_contact, &sr)
#
#            # calculate interface flux - eq. 10.71
#            if(wn <= sl):
#
#                # left state
#                fm.data[i]  = _dl*(Vnl - wn)
#                fe.data[i]  = (0.5*_dl*vl_sq + _pl/(gamma - 1.0))*(Vnl - wn) + _pl*Vnl
#
#                for k in range(dim):
#                    fmv[k][i] = _dl*vl[k][i]*(Vnl - wn) + _pl*nx[k][i]
#
#            elif((sl < wn) and (wn <= sr)):
#
#                # intermediate state
#                if(wn <= s_contact):
#
#                    # left star state - eq. 10.38 and 10.39
#                    factor_1 = _dl*(sl - Vnl)/(sl - s_contact)
#                    factor_2 = factor_1*(sl - wn)*(s_contact - Vnl) + _pl
#                    frho = _dl*(Vnl - sl) + factor_1*(sl - wn)
#
#                    # total energy
#                    el = 0.5*_dl*vl_sq + _pl/(gamma-1.0)
#
#                    fm.data[i] = frho
#                    fe.data[i] = (el + _pl)*Vnl - sl*el +\
#                            (sl - wn)*factor_1*(el/_dl + (s_contact - Vnl)*\
#                            (s_contact + _pl/(_dl*(sl - Vnl))))
#
#                    for k in range(dim):
#                        fmv[k][i] = frho*vl[k][i] + factor_2*nx[k][i]
#
#                else:
#
#                    # right star state
#                    factor_1 = _dr*(sr - Vnr)/(sr - s_contact)
#                    factor_2 = factor_1*(sr - wn)*(s_contact - Vnr) + _pr
#                    frho = _dr*(Vnr - sr) + factor_1*(sr - wn)
#
#                    # total energy
#                    er = 0.5*_dr*vr_sq + _pr/(gamma-1.0)
#
#                    fm.data[i] = frho
#                    fe.data[i] = (er + _pr)*Vnr - sr*er +\
#                            (sr - wn)*factor_1*(er/_dr + (s_contact - Vnr)*\
#                            (s_contact + _pr/(_dr*(sr - Vnr))))
#
#                    for k in range(dim):
#                        fmv[k][i] = frho*vr[k][i] + factor_2*nx[k][i]
#
#            else:
#
#                # right state
#                fm.data[i]  = _dr*(Vnr - wn)
#                fe.data[i]  = (0.5*_dr*vr_sq + _pr/(gamma - 1.0))*(Vnr - wn) + _pr*Vnr
#
#                for k in range(dim):
#                    fmv[k][i] = _dr*vr[k][i]*(Vnr - wn) + _pr*nx[k][i]
#
#        if boost == 1:
#            self.deboost(fluxes, faces, dim)
#
#cdef class Exact(RiemannBase):
#    def __init__(self ):
#        # self.reconstruction = None
#        pass
#
#    cdef riemann_solver(self, CarrayContainer fluxes, CarrayContainer left_faces, CarrayContainer right_faces,
#            CarrayContainer faces, double gamma, int dim):
#
#        # left state primitive variables
#        cdef DoubleArray dl = left_faces.get_carray("density")
#        cdef DoubleArray pl = left_faces.get_carray("pressure")
#
#        # left state primitive variables
#        cdef DoubleArray dr = right_faces.get_carray("density")
#        cdef DoubleArray pr = right_faces.get_carray("pressure")
#
#        cdef DoubleArray fm  = fluxes.get_carray("mass")
#        cdef DoubleArray fe  = fluxes.get_carray("energy")
#
#        cdef int i, k
#        cdef np.float64_t *vl[3], *vr[3], *fmv[3], *nx[3]
#        cdef np.float64_t _vl[3], _vr[3], n[3]
#
#        # state values
#        cdef double _dl, _pl
#        cdef double _dr, _pr
#        cdef double _d, v[3], _p, vn, v_sq
#
#        cdef double vnl, vnr, vl_sq, vr_sq
#
#        # wave estimates
#        cdef double fr, fl, u_tmp
#        cdef double p_star, u_star
#        cdef double s_hl, s_tl, sl, sr
#        cdef double c, cl, cr, c_star_l, c_star_r
#
#        cdef int num_faces = faces.get_number_of_items()
#
#        left_faces.pointer_groups(vl,  left_faces.named_groups['velocity'])
#        right_faces.pointer_groups(vr, right_faces.named_groups['velocity'])
#        fluxes.pointer_groups(fmv, fluxes.named_groups['momentum'])
#        faces.pointer_groups(nx, faces.named_groups['normal'])
#
#        for i in range(num_faces):
#
#            # left state
#            _dl = dl.data[i]
#            _pl = pl.data[i]
#
#            # right state
#            _dr = dr.data[i]
#            _pr = pr.data[i]
#
#            # sound speed
#            cl = sqrt(gamma*_pl/_dl)
#            cr = sqrt(gamma*_pr/_dr)
#
#            vnl = vnr = 0.0
#            vl_sq = vr_sq = 0.0
#            for k in range(dim):
#
#                _vl[k] = vl[k][i]
#                _vr[k] = vr[k][i]
#                n[k]   = nx[k][i]
#
#                # project left/righ velocity to face normal
#                vnl += _vl[k]*n[k]
#                vnr += _vr[k]*n[k]
#
#                # left/right velocity square
#                vl_sq += _vl[k]*_vl[k]
#                vr_sq += _vr[k]*_vr[k]
#
#            # hack - delete later >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#            if _dl < 0. or _dr < 0.:
#
#                print 'vacuum left/right'
#
#                self.vacuum(_dl, _vl, _pl, vnl, cl,\
#                       _dr, _vr, _pr, vnr, cr,\
#                       &_d,  v, &_p, &vn, &v_sq,\
#                       gamma, n, dim)
#
#                fm.data[i] = _d*vn
#                fe.data[i] = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn
#                for k in range(dim):
#                    fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]
#
#                continue
#
#            if (2.*cl/(gamma-1.) + 2.*cr/(gamma-1.)) <= (vnr - vnl):
#
#                print 'vacuum generation'
#
#                self.vacuum(_dl, _vl, _pl, vnl, cl,\
#                       _dr, _vr, _pr, vnr, cr,\
#                       &_d,  v, &_p, &vn, &v_sq,\
#                       gamma, n, dim)
#
#                fm.data[i] = _d*vn
#                fe.data[i] = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn
#                for k in range(dim):
#                    fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]
#
#                continue
#            # hack - delete later <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
#
#            # newton rhapson 
#            p_star = self.get_pstar(_dl, vnl, _pl, cl,
#                    _dr, vnr, _pr, cr, gamma)
#
#            # calculate the contact wave speed
#            fl = self.p_func(_dl, vnl, _pl, cl, gamma, p_star)
#            fr = self.p_func(_dr, vnr, _pr, cr, gamma, p_star)
#            u_star = 0.5*(vnl + vnr + fr - fl)
#
#            if(0.0 <= u_star): # left of contact discontinuity
#                if(p_star <= _pl): # left rarefraction
#
#                    # sound speed of head
#                    s_hl = vnl - cl
#
#                    if(0.0 <= s_hl): # left state
#                        fm.data[i]  = _dl*vnl
#                        fe.data[i]  = (0.5*_dl*vl_sq + gamma*_pl/(gamma - 1.0))*vnl
#
#                        for k in range(dim):
#                            fmv[k][i] = _dl*vl[k][i]*vnl + _pl*nx[k][i]
#
#                    else: # left rarefaction
#
#                        # sound speed of star state and tail of rarefraction
#                        c_star_l = cl*pow(p_star/_pl, (gamma - 1.0)/(2.0*gamma))
#                        s_tl = u_star - c_star_l
#
#                        if(0.0 >= s_tl): # star left state
#                            _d = _dl*pow(p_star/_pl, 1.0/gamma)
#                            _p = p_star
#
#                            v_sq = vn = 0.
#                            for k in range(dim):
#                                v[k]  = vl[k][i] + (u_star - vnl)*nx[k][i]
#                                vn   += v[k]*nx[k][i]
#                                v_sq += v[k]*v[k]
#
#                            fm.data[i]  = _d*vn
#                            fe.data[i]  = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn
#
#                            for k in range(dim):
#                                fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]
#
#                        else: # inside left fan
#
#                            c  = (2.0/(gamma + 1.0))*(cl + 0.5*(gamma - 1.0)*vnl)
#                            _d = _dl*pow(c/cl, 2.0/(gamma - 1.0))
#                            _p = _pl*pow(c/cl, 2.0*gamma/(gamma - 1.0))
#
#                            v_sq = vn = 0.
#                            for k in range(dim):
#                                v[k]  = vl[k][i] + (c - vnl)*nx[k][i]
#                                vn   += v[k]*nx[k][i]
#                                v_sq += v[k]*v[k]
#
#                            fm.data[i] = _d*vn
#                            fe.data[i] = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn
#
#                            for k in range(dim):
#                                fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]
#
#                else: # left shock
#
#                    sl = vnl - cl*sqrt((gamma + 1.0)*p_star/(2.0*gamma*_pl)\
#                            + (gamma - 1.0)/(2.0*gamma))
#
#                    if(0.0 <= sl): # left state
#                        fm.data[i] = _dl*vnl
#                        fe.data[i] = (0.5*_dl*vl_sq + gamma*_pl/(gamma - 1.0))*vnl
#
#                        for k in range(dim):
#                            fmv[k][i] = _dl*vl[k][i]*vnl + _pl*nx[k][i]
#
#                    else: # star left state
#
#                        _d = _dl*(p_star/_pl + (gamma - 1.0)/(gamma + 1.0))\
#                                /(p_star*(gamma - 1.0)/((gamma + 1.0)*_pl) + 1.0)
#                        _p = p_star
#
#                        v_sq = vn = 0.
#                        for k in range(dim):
#                            v[k]  = vl[k][i] + (u_star - vnl)*nx[k][i]
#                            vn   += v[k]*nx[k][i]
#                            v_sq += v[k]*v[k]
#
#                        fm.data[i]  = _d*vn
#                        fe.data[i]  = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn
#
#                        for k in range(dim):
#                            fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]
#
#            else: # right of contact
#
#                if(p_star >= _pr): # right shock
#
#                    sr = vnr + cr*sqrt((gamma + 1.0)*p_star/(2.0*gamma*_pr)\
#                            + (gamma-1.0)/(2.0*gamma))
#
#                    if(0.0 >= sr): # right state
#                        fm.data[i] = _dr*vnr
#                        fe.data[i] = (0.5*_dr*vr_sq + gamma*_pr/(gamma - 1.0))*vnr
#
#                        for k in range(dim):
#                            fmv[k][i] = _dr*vr[k][i]*vnr + _pr*nx[k][i]
#
#                    else: # star right state
#
#                        _d = _dr*(p_star/_pr + (gamma - 1.0)/(gamma + 1.0))\
#                                /(p_star*(gamma - 1.0)/((gamma + 1.0)*_pr) + 1.0)
#                        _p = p_star
#
#                        v_sq = vn = 0.
#                        for k in range(dim):
#                            v[k]  = vr[k][i] + (u_star - vnr)*nx[k][i]
#                            vn   += v[k]*nx[k][i]
#                            v_sq += v[k]*v[k]
#
#                        fm.data[i]  = _d*vn
#                        fe.data[i]  = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn
#
#                        for k in range(dim):
#                            fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]
#
#                else: # right rarefaction
#
#                    s_hr = vnr + cr
#
#                    if(0.0 >= s_hr): # right data state
#                        fm.data[i]  = _dr*vnr
#                        fe.data[i]  = (0.5*_dr*vr_sq + gamma*_pr/(gamma - 1.0))*vnr
#
#                        for k in range(dim):
#                            fmv[k][i] = _dr*vr[k][i]*vnr + _pr*nx[k][i]
#
#                    else:
#
#                        # sound speed of the star state and sound speed
#                        # of the tail of the rarefraction
#                        c_star_r = cr*pow(p_star/_pr, (gamma-1.0)/(2.0*gamma))
#                        s_tr = u_star + c_star_r
#
#                        if(0.0 <= s_tr): # star left state
#                            _d = _dr*pow(p_star/_pr, 1.0/gamma)
#                            _p = p_star
#
#                            v_sq = vn = 0.
#                            for k in range(dim):
#                                v[k]  = vr[k][i] + (u_star - vnr)*nx[k][i]
#                                vn   += v[k]*nx[k][i]
#                                v_sq += v[k]*v[k]
#
#                            fm.data[i]  = _d*vn
#                            fe.data[i]  = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn
#
#                            for k in range(dim):
#                                fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]
#
#                        else:
#
#                            # sampled point is inside right fan
#                            c = (2.0/(gamma + 1.0))*(cr - 0.5*(gamma - 1.0)*vnr)
#                            u_tmp = (2.0/(gamma + 1.0))*(-cr + 0.5*(gamma-1.0)*vnr)
#
#                            _d = _dr*pow(c/cr, 2.0/(gamma - 1.0))
#                            _p = _pr*pow(c/cr, 2.0*gamma/(gamma - 1.0))
#
#                            v_sq = vn = 0.
#                            for k in range(dim):
#                                v[k]  = vr[k][i] + (u_tmp - vnr)*nx[k][i]
#                                vn   += v[k]*nx[k][i]
#                                v_sq += v[k]*v[k]
#
#                            fm.data[i]  = _d*vn
#                            fe.data[i]  = (0.5*_d*v_sq + gamma*_p/(gamma - 1.0))*vn
#
#                            for k in range(dim):
#                                fmv[k][i] = _d*v[k]*vn + _p*nx[k][i]
#
#        self._deboost(fluxes, faces, dim)
#
#    @cython.cdivision(True)
#    cdef inline double p_guess(self, double dl, double ul, double pl, double cl,
#            double dr, double ur, double pr, double cr, double gamma) nogil:
#        """
#        Calculate  starting pressure for iterative exact scheme
#        Reference: Toro (2009): Chapter 4
#        """
#        cdef double ppv
#        cdef double p_lr, p_tl
#        cdef double gl, gr, p0
#        cdef double p_star, p_max, p_min, q_max
#
#        # initial guess for pressure eq: 4.47
#        ppv = .5*(pl + pr) - .125*(ur - ul)*(dl + dr)*(cl + cr)
#
#        p_star = max(0., ppv)
#        p_max  = max(pl, pr)
#        p_min  = min(pl, pr)
#        q_max  = p_max/p_min
#
#        if ((q_max <= 2.) and (p_min <= ppv <= p_max)):
#            p0 = ppv
#
#        elif (ppv <= p_min):
#            p_lr   = pow(pl/pr, (gamma - 1.)/(2.*gamma))
#            u_star = (p_lr*ul/cl + ur/cr + 2.*(p_lr - 1.)/(gamma - 1.))
#            u_star = u_star/(p_lr/cl + 1./cr)
#            p_tl   = pow(1. + (gamma - 1.)*(ul - u_star)/(2.*cl), 2.*gamma/(gamma - 1.))
#            p_tr   = pow(1. + (gamma - 1.)*(u_star - ur)/(2.*cr), 2.*gamma/(gamma - 1.))
#            p0 = .5*(pl*p_tl + pr*p_tr)
#
#        else:
#            gl = sqrt((2./(dl*(gamma + 1.)))/((gamma - 1.)*pl/(gamma + 1.) + ppv))
#            gr = sqrt((2./(dr*(gamma + 1.)))/((gamma - 1.)*pr/(gamma + 1.) + ppv))
#            p0 = (gl*pl + gr*pr - (ur - ul))/(gr + gl)
#
#        return p0
#
#    @cython.cdivision(True)
#    cdef inline double p_func(self, double d, double u, double p,
#            double c, double gamma, double p_old) nogil:
#        """
#        Calculate the derivative of the jump across the wave.
#        Reference: Toro (2009): Chapter 4
#        """
#        cdef double f, Ak, Bk
#
#        # rarefaction wave eq: 4.6b and 4.7b
#        if (p_old <= p):
#            f = 2.*c/(gamma - 1.)*(pow(p_old/p, (gamma - 1.)/(2.*gamma)) - 1.)
#
#        # shock wave eq: 4.6a and 4.7a
#        else:
#            Ak = 2./(d*(gamma + 1.))
#            Bk = p*(gamma - 1.)/(gamma + 1.)
#            f = (p_old - p)*sqrt(Ak/(p_old + Bk))
#
#        return f
#
#    @cython.cdivision(True)
#    cdef inline double p_func_deriv(self, double d, double u, double p,
#            double c, double gamma, double p_old) nogil:
#        """
#        Calculate the derivative of the jump across the wave.
#        Reference: Toro (2009): Chapter 4
#        """
#        cdef double df, Ak, Bk
#
#        # derivative for rarefaction wave eq. 4.37
#        if (p_old <= p):
#            df = pow(p_old/p, -(gamma + 1.)/(2.*gamma))/(c*d)
#
#        # derivative for shock wave
#        else:
#            # eq: 4.8 and 4.37
#            Ak = 2./(d*(gamma + 1.))
#            Bk = p*(gamma - 1.)/(gamma + 1.)
#            df = sqrt(Ak/(p_old + Bk))*(1. - .5*(p_old - p)/(Bk + p_old))
#
#        return df
#
#    @cython.cdivision(True)
#    cdef inline double get_pstar(self, double dl, double ul, double pl, double cl,
#            double dr, double ur, double pr, double cr, double gamma) nogil:
#        """
#        Calculate star pressure by iteration.
#        Reference: Toro (2009): Chapter 4
#        """
#        cdef double TOL = 1.0e-6
#        cdef int MAX_ITER = 1000
#
#        cdef int i = 0
#        cdef double p_old, p_new, p
#        cdef double fr, fl, df_r, df_l
#
#        p_old = self.p_guess(dl, ul, pl, cl, dr, ur, pr, cr, gamma)
#        while(i < MAX_ITER):
#
#            fl  = self.p_func(dl, ul, pl, cl, gamma, p_old)
#            fr  = self.p_func(dr, ur, pr, cr, gamma, p_old)
#            dfl = self.p_func_deriv(dl, ul, pl, cl, gamma, p_old)
#            dfr = self.p_func_deriv(dr, ur, pr, cr, gamma, p_old)
#
#            p_new = p_old - (fl + fr + ur - ul)/(dfl + dfr)
#
#            if ( 2.*fabs((p_new - p_old)/(p_new + p_old)) ) <= TOL:
#                return p_new
#
#            if (p_new < 0.):
#                p_new = TOL
#
#            p_old = p_new
#            i += 1
#
#        # failed to converge
#        with gil:
#            raise RuntimeError('No convergence in Exact Riemann Solver')
#
#    cdef inline void vacuum(self,
#            double dl, double vl[3], double pl, double vnl, double cl,
#            double dr, double vr[3], double pr, double vnr, double cr,
#            double *d, double  v[3], double *p, double *vn, double *vsq,
#            double gamma, double n[3], int dim) nogil:
#        """
#        Calculate vacuum solution.
#        Reference: Toro (2009): Chapter 4
#        """
#        cdef int i
#        cdef double c, u_tmp
#
#        cdef double sl = vnl + 2.*cl/(gamma-1.)
#        cdef double sr = vnr - 2.*cr/(gamma-1.)
#
#        if(dr < 0): # right vacuum eq 4.77
#            if(0. <= vnl - cl): # left state 
#                d[0] = dl
#                p[0] = pl
#
#                vn[0] = vsq[0] = 0.
#                for i in range(dim):
#                    v[i]    = vl[i]
#                    vn[0]  += v[i]*n[i]
#                    vsq[0] += v[i]*v[i]
#
#            elif(0. < sl): # left fan
#                c    = (2./(gamma + 1.))*(cl + .5*(gamma - 1.)*vnl)
#                d[0] = dl*pow(c/cl, 2./(gamma - 1.))
#                p[0] = pl*pow(c/cl, 2.*gamma/(gamma - 1.))
#
#                vn[0] = vsq[0] = 0.
#                for i in range(dim):
#                    v[i]    = vl[i] + (c - vnl)*n[i]
#                    vn[0]  += v[i]*n[i]
#                    vsq[0] += v[i]*v[i]
#
#            else: # right vacuum
#                d[0] = 0.
#                p[0] = 0.
#
#                vn[0] = vsq[0] = 0
#                for i in range(dim):
#                    v[i] = 0.
#
#        elif(dl < 0): # left vacuum
#            if 0. <= sr: # left vacuum
#                d[0] = 0.
#                p[0] = 0.
#
#                vn[0] = vsq[0] = 0.
#                for i in range(dim):
#                    v[i] = 0.
#
#            elif(0. < vnr + cr): # right fan
#                # sampled point is inside right fan
#                c = (2./(gamma + 1.))*(cr - .5*(gamma - 1.)*vnr)
#                u_tmp = (2./(gamma + 1.))*(-cr + .5*(gamma-1.)*vnr)
#
#                d[0] = dr*pow(c/cr, 2./(gamma - 1.))
#                p[0] = pr*pow(c/cr, 2.*gamma/(gamma - 1.))
#
#                vn[0] = vsq[0] = 0
#                for i in range(dim):
#                    v[i]    = vr[i] + (u_tmp - vnr)*n[i]
#                    vn[0]  += v[i]*n[i]
#                    vsq[0] += v[i]*v[i]
#
#            else: # right state
#                d[0] = dr
#                p[0] = pr
#
#                vn[0] = vsq[0] = 0
#                for i in range(dim):
#                    v[i]    = vr[i]
#                    vn[0]  += v[i]*n[i]
#                    vsq[0] += v[i]*v[i]
#
#        else: # vacuum generation
#
#            if(sl < 0.) and (0. < sr): # vacuum
#                d[0] = 0.
#                p[0] = 0.
#
#                vn[0] = vsq[0] = 0
#                for i in range(dim):
#                    v[i] = 0.
#
#            elif(0. <= sl):
#                if 0. <= vnl - cl: # left state 
#                    d[0] = dl
#                    p[0] = pl
#
#                    vn[0] = vsq[0] = 0.
#                    for i in range(dim):
#                        v[i]    = vl[i]
#                        vn[0]  += v[i]*n[i]
#                        vsq[0] += v[i]*v[i]
#
#                else: # left fan
#                    c    = (2./(gamma + 1.))*(cl + .5*(gamma - 1.)*vnl)
#                    d[0] = dl*pow(c/cl, 2./(gamma - 1.))
#                    p[0] = pl*pow(c/cl, 2.*gamma/(gamma - 1.))
#
#                    vn[0] = vsq[0] = 0.
#                    for i in range(dim):
#                        v[i]    = vl[i] + (c - vnl)*n[i]
#                        vn[0]  += v[i]*n[i]
#                        vsq[0] += v[i]*v[i]
#
#            else:
#                if(0. < vnr + cr): # right fan
#                    # sample inside right fan
#                    c = (2./(gamma + 1.))*(cr - .5*(gamma - 1.)*vnr)
#                    u_tmp = (2./(gamma + 1.))*(-cr + .5*(gamma-1.)*vnr)
#
#                    d[0] = dr*pow(c/cr, 2./(gamma - 1.))
#                    p[0] = pr*pow(c/cr, 2.*gamma/(gamma - 1.))
#
#                    vn[0] = vsq[0] = 0.
#                    for i in range(dim):
#                        v[i]    = vr[i] + (u_tmp - vnr)*n[i]
#                        vn[0]  += v[i]*n[i]
#                        vsq[0] += v[i]*v[i]
#
#                else: # right state
#                    d[0] = dr
#                    p[0] = pr
#
#                    vn[0] = vsq[0] = 0.
#                    for i in range(dim):
#                        v[i]    = vr[i]
#                        vn[0]  += v[i]*n[i]
#                        vsq[0] += v[i]*v[i]
#
