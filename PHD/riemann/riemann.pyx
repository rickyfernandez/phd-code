from libc.math cimport sqrt, pow, fabs
import numpy as np
cimport numpy as np
cimport cython

#cdef hllc_state(double[:] u_str, double[:] q, double gamma, double s_wave, double contact_wave):
#
#    cdef double factor = q[0]*(s_wave - q[1])/(s_wave - contact_wave)
#
#    # star state - eq. 10.39
#    u_str[0] = factor
#    u_str[1] = factor*contact_wave
#    u_str[2] = factor*q[2];
#    u_str[3] = factor*(0.5*(q[1]*q[1] + q[2]*q[2]) + q[3]/(q[0]*(gamma - 1.0)) +\
#            (contact_wave - q[1])*(contact_wave + q[3]/(q[0]*(s_wave - q[1]))))
#
#cdef construct_flux(double[:] flux_state, double[:] q, double gamma):
#
#    flux_state[0] = q[0]*q[1]
#    flux_state[1] = q[0]*q[1]*q[1] + q[3]
#    flux_state[2] = q[0]*q[1]*q[2]
#    flux_state[3] = q[1]*(0.5*q[0]*(q[1]*q[1] + q[2]*q[2]) + q[3]*gamma/(gamma - 1.0))
#
#def hll(double[:,::1] face_l, double[:,::1] face_r, double[:,::1] flux, double[:] w_face, double gamma, int num_faces):
#
#    cdef double w
#    cdef int i, j
#    cdef double s_l, s_r, s_contact
#    cdef double d_l, p_l, u_l
#    cdef double d_r, p_r, u_r
#
#    cdef double[:] flux_state = np.zeros(4, dtype=np.float64)
#    cdef double[:] flux_l     = np.zeros(4, dtype=np.float64)
#    cdef double[:] flux_r     = np.zeros(4, dtype=np.float64)
#
#    cdef double[:] u_state_l = np.zeros(4, dtype=np.float64)
#    cdef double[:] q_l       = np.zeros(4, dtype=np.float64)
#
#    cdef double[:] u_state_r = np.zeros(4, dtype=np.float64)
#    cdef double[:] q_r       = np.zeros(4, dtype=np.float64)
#
#    cdef double[:] waves = np.zeros(3, dtype=np.float64)
#
#    for i in range(num_faces):
#
#        d_l = face_l[0,i]
#        u_l = face_l[1,i]
#        p_l = face_l[3,i]
#
#        d_r = face_r[0,i]
#        u_r = face_r[1,i]
#        p_r = face_r[3,i]
#
#        get_waves(d_l, u_l, p_l, d_r, u_r, p_r, gamma, waves)
#        s_l = waves[0]; s_contact = waves[1]; s_r = waves[2]
#
#        for j in range(4):
#
#            q_l[j]  = face_l[j,i]
#            q_r[j] = face_r[j,i]
#
#        # convert from primitive variables to conserative variables
#        prim_to_cons(u_state_l, q_l, gamma)
#        prim_to_cons(u_state_r, q_r, gamma)
#
#        w = w_face[i]
#
#        # calculate interface flux - eq. 10.71
#        if(w <= s_l):
#            # l state
#            construct_flux(flux_state, q_l, gamma)
#
#            for j in range(4):
#                flux_state[j] -= w*u_state_l[j]
#
#        elif((s_l < w) and (w <= s_r)):
#
#            construct_flux(flux_l, q_l, gamma)
#            construct_flux(flux_r, q_r, gamma)
#
#            for j in range(4):
#                flux_state[j]  = 0.0
#                flux_state[j] += (s_r*flux_l[j] - s_l*flux_r[j] + s_l*s_r*(u_state_r[j] - u_state_l[j]))/(s_r - s_l)
#                flux_state[j] -= w*(s_r*u_state_r[j] - s_l*u_state_l[j] + flux_l[j] - flux_r[j])/(s_r - s_l)
#
#        else:
#
#            # it's a right state
#            construct_flux(flux_state, q_r, gamma)
#
#            for j in range(4):
#                flux_state[j] -= w*u_state_r[j]
#
#        for j in range(4):
#            flux[j,i] = flux_state[j]
#
#def hllc(double[:,::1] face_l, double[:,::1] face_r, double[:,::1] flux, double[:] w_face, double gamma, int num_faces):
#
#    cdef double w
#    cdef int i, j
#    cdef double s_l, s_r, s_contact
#    cdef double d_l, p_l, u_l
#    cdef double d_r, p_r, u_r
#
#    cdef double[:] flux_state = np.zeros(4, dtype=np.float64)
#    cdef double[:] u_state_l = np.zeros(4, dtype=np.float64)
#    cdef double[:] q_l = np.zeros(4, dtype=np.float64)
#    cdef double[:] u_star_l = np.zeros(4, dtype=np.float64)
#    cdef double[:] u_state_r = np.zeros(4, dtype=np.float64)
#    cdef double[:] q_r = np.zeros(4, dtype=np.float64)
#    cdef double[:] u_star_r = np.zeros(4, dtype=np.float64)
#    cdef double[:] waves  = np.zeros(3, dtype=np.float64)
#
#    for i in range(num_faces):
#
#        d_l = face_l[0,i]
#        u_l = face_l[1,i]
#        p_l = face_l[3,i]
#
#        d_r = face_r[0,i]
#        u_r = face_r[1,i]
#        p_r = face_r[3,i]
#
#        get_waves(d_l, u_l, p_l, d_r, u_r, p_r, gamma, waves)
#        s_l= waves[0]; s_contact = waves[1]; s_r = waves[2]
#
#        for j in range(4):
#
#            q_l[j] = face_l[j,i]
#            q_r[j] = face_r[j,i]
#
#        # convert from primitive variables to conserative variables
#        prim_to_cons(u_state_l, q_l, gamma)
#        prim_to_cons(u_state_r, q_r, gamma)
#
#        w = w_face[i]
#
#        # calculate interface flux - eq. 10.71
#        if(w <= s_l):
#            # left state
#            construct_flux(flux_state, q_l, gamma)
#
#            for j in range(4):
#                flux_state[j] -= w*u_state_l[j]
#
#        elif((s_l < w) and (w <= s_r)):
#
#            if(w <= s_contact):
#
#                hllc_state(u_star_l, q_l, gamma, s_l, s_contact)
#                construct_flux(flux_state, q_l, gamma)
#
#                for j in range(4):
#                    flux_state[j] += s_l*(u_star_l[j] - u_state_l[j]) - w*u_star_l[j]
#
#            else:
#
#                hllc_state(u_star_r, q_r, gamma, s_r, s_contact)
#                construct_flux(flux_state, q_r, gamma)
#
#                for j in range(4):
#                    flux_state[j] += s_r*(u_star_r[j] - u_state_r[j]) - w*u_star_r[j]
#
#        else:
#
#            # it's a right state
#            construct_flux(flux_state, q_r, gamma)
#
#            for j in range(4):
#                flux_state[j] -= w*u_state_r[j]
#
#        for j in range(4):
#            flux[j,i] = flux_state[j]
#
#def exact(double[:,::1] face_left, double[:,::1] face_right, double[:,::1] face_state, double gamma, int num_faces):
#
#    cdef double d_l, u_l, v_l, p_l
#    cdef double d_r, u_r, v_r, p_r
#    cdef double d, u, v, p
#
#    cdef double p_star, u_star
#    cdef double f_r, f_l
#    cdef double s_hl, s_tl, s_l, s_r, c
#    cdef double c_l, c_r, c_star_l, c_star_r
#
#    for i in range(num_faces):
#
#        d_l = face_left[0,i]
#        u_l = face_left[1,i]
#        v_l = face_left[2,i]
#        p_l = face_left[3,i]
#
#        d_r = face_right[0,i]
#        u_r = face_right[1,i]
#        v_r = face_right[2,i]
#        p_r = face_right[3,i]
#
#        c_l = sqrt(gamma*p_l/d_l)
#        c_r = sqrt(gamma*p_r/d_r)
#
#        # newton rhapson 
#        p_star = get_pstar(d_l, u_l, p_l, d_r, u_r, p_r, gamma)
#
#        # calculate the contact wave speed
#        f_r = p_func(d_r, u_r, p_r, gamma, p_star)
#        f_l = p_func(d_l, u_l, p_l, gamma, p_star)
#        u_star = 0.5*(u_l + u_r + f_r - f_l)
#
#        if(0.0 <= u_star):
#
#            # transverse velocity only jumps across the contact
#            v = v_l
#
#            # left of the contact discontinuity
#            if(p_star <= p_l):
#
#                # left rarefraction, below is the sound speed of the head
#                s_hl = u_l - c_l
#
#                if(0.0 <= s_hl):
#
#                    # sample point is left data state
#                    d = d_l
#                    u = u_l
#                    p = p_l
#
#                else:
#
#                    # sound speed of the star state and sound speed
#                    # of the tail of the rarefraction
#                    c_star_l = c_l*pow(p_star/p_l, (gamma - 1.0)/(2.0*gamma))
#                    s_tl = u_star - c_star_l
#
#                    if(0.0 >= s_tl):
#
#                        # sample point is star left state
#                        d = d_l*pow(p_star/p_l, 1.0/gamma)
#                        u = u_star
#                        p = p_star
#
#                    else:
#
#                        # sampled point is inside left fan
#                        c = (2.0/(gamma + 1.0))*(c_l + 0.5*(gamma - 1.0)*u_l)
#
#                        d = d_l*pow(c/c_l, 2.0/(gamma - 1.0))
#                        u = c
#                        p = p_l*pow(c/c_l, 2.0*gamma/(gamma - 1.0))
#            else:
#
#                # left shock
#                s_l = u_l - c_l*sqrt((gamma + 1.0)*p_star/(2.0*gamma*p_l) + (gamma - 1.0)/(2.0*gamma))
#
#                if(0.0 <= s_l):
#
#                    # sampled point is a left data state
#                    d = d_l
#                    u = u_l
#                    p = p_l
#
#                else:
#
#                    # sampled point is star left state
#                    d = d_l*(p_star/p_l + (gamma - 1.0)/(gamma + 1.0))/(p_star*(gamma - 1.0)/((gamma + 1.0)*p_l) + 1.0)
#                    u = u_star
#                    p = p_star
#
#
#        else:
#
#            # transverse velocity only jumps across the contact
#            v = v_r
#
#            # sampling point lies to the right of the contact
#            if(p_star >= p_r):
#
#                # right shock
#                s_r = u_r + c_r*sqrt((gamma + 1.0)*p_star/(2.0*gamma*p_r) + (gamma-1.0)/(2.0*gamma))
#
#                if(0.0 >= s_r):
#
#                    # sampled point is right data states
#                    d = d_r
#                    u = u_r
#                    p = p_r
#
#                else:
#
#                    # sample point is star right state
#                    d = d_r*(p_star/p_r + (gamma - 1.0)/(gamma + 1.0))/(p_star*(gamma - 1.0)/((gamma + 1.0)*p_r) + 1.0)
#                    u = u_star
#                    p = p_star
#
#            else:
#
#                # right rarefaction
#                s_hr = u_r + c_r
#
#                if(0.0 >= s_hr):
#
#                    # sample point is right data state
#                    d = d_r
#                    u = u_r
#                    p = p_r
#
#                else:
#
#                    # sound speed of the star state and sound speed
#                    # of the tail of the rarefraction
#                    c_star_r = c_r*pow(p_star/p_r, (gamma-1.0)/(2.0*gamma))
#                    s_tr = u_star + c_star_r
#
#                    if(0.0 <= s_tr):
#
#                        # sample point is star left state
#                        d = d_r*pow(p_star/p_r, 1.0/gamma)
#                        u = u_star
#                        p = p_star
#
#                    else:
#
#                        # sampled point is inside right fan
#                        c = (2.0/(gamma + 1.0))*(c_r - 0.5*(gamma - 1.0)*u_r)
#
#                        d = d_r*pow(c/c_r, 2.0/(gamma - 1.0))
#                        u = (2.0/(gamma + 1.0))*(-c_r + 0.5*(gamma-1.0)*u_r)
#                        p = p_r*pow(c/c_r, 2.0*gamma/(gamma - 1.0))
#
#        face_state[0,i] = d
#        face_state[1,i] = u
#        face_state[2,i] = v
#        face_state[3,i] = p

def exact(double[:,::1] face_l, double[:,::1] face_r, double[:,::1] fluxes, double[:,::1] normal, double[:,::1] w_face, double gamma, int num_faces):

    cdef double d_l, u_l, v_l, p_l
    cdef double d_r, u_r, v_r, p_r
    cdef double d, u, v, p

    cdef double un_l, un_r
    cdef double wx, wy

    cdef double u_tmp

    cdef double p_star, u_star
    cdef double f_r, f_l
    cdef double s_hl, s_tl, s_l, s_r, c
    cdef double c_l, c_r, c_star_l, c_star_r

    cdef double[:] n = np.zeros(2, dtype=np.float64)

    for i in range(num_faces):

        d_l = face_l[0,i]
        u_l = face_l[1,i]
        v_l = face_l[2,i]
        p_l = face_l[3,i]

        d_r = face_r[0,i]
        u_r = face_r[1,i]
        v_r = face_r[2,i]
        p_r = face_r[3,i]

        c_l = sqrt(gamma*p_l/d_l)
        c_r = sqrt(gamma*p_r/d_r)

        # the norm of the face
        n[0] = normal[0,i]
        n[1] = normal[1,i]

        # project velocity onto face normal
        un_l = u_l*n[0] + v_l*n[1]
        un_r = u_r*n[0] + v_r*n[1]

        # newton rhapson 
        p_star = get_pstar(d_l, un_l, p_l, d_r, un_r, p_r, gamma)

        # calculate the contact wave speed
        f_l = p_func(d_l, un_l, p_l, gamma, p_star)
        f_r = p_func(d_r, un_r, p_r, gamma, p_star)
        u_star = 0.5*(un_l + un_r + f_r - f_l)

        if(0.0 <= u_star):

            # transverse velocity only jumps across the contact
            #v = v_l

            # left of the contact discontinuity
            if(p_star <= p_l):

                # left rarefraction, below is the sound speed of the head
                s_hl = un_l - c_l

                if(0.0 <= s_hl):

                    # sample point is left data state
                    d = d_l
                    u = u_l
                    v = v_l
                    p = p_l

                else:

                    # sound speed of the star state and sound speed
                    # of the tail of the rarefraction
                    c_star_l = c_l*pow(p_star/p_l, (gamma - 1.0)/(2.0*gamma))
                    s_tl = u_star - c_star_l

                    if(0.0 >= s_tl):

                        # sample point is star left state
                        d = d_l*pow(p_star/p_l, 1.0/gamma)
                        u = u_l + (u_star - un_l)*n[0]
                        v = v_l + (u_star - un_l)*n[1]
                        p = p_star

                    else:

                        # sampled point is inside left fan
                        c = (2.0/(gamma + 1.0))*(c_l + 0.5*(gamma - 1.0)*un_l)

                        d = d_l*pow(c/c_l, 2.0/(gamma - 1.0))
                        u = u_l + (c - un_l)*n[0]
                        v = v_l + (c - un_l)*n[1]
                        p = p_l*pow(c/c_l, 2.0*gamma/(gamma - 1.0))
            else:

                # left shock
                s_l = un_l - c_l*sqrt((gamma + 1.0)*p_star/(2.0*gamma*p_l) + (gamma - 1.0)/(2.0*gamma))

                if(0.0 <= s_l):

                    # sampled point is a left data state
                    d = d_l
                    u = u_l
                    v = v_l
                    p = p_l

                else:

                    # sampled point is star left state
                    d = d_l*(p_star/p_l + (gamma - 1.0)/(gamma + 1.0))/(p_star*(gamma - 1.0)/((gamma + 1.0)*p_l) + 1.0)
                    u = u_l + (u_star - un_l)*n[0]
                    v = v_l + (u_star - un_l)*n[1]
                    p = p_star


        else:

            # transverse velocity only jumps across the contact
            #v = v_r

            # sampling point lies to the right of the contact
            if(p_star >= p_r):

                # right shock
                s_r = un_r + c_r*sqrt((gamma + 1.0)*p_star/(2.0*gamma*p_r) + (gamma-1.0)/(2.0*gamma))

                if(0.0 >= s_r):

                    # sampled point is right data states
                    d = d_r
                    u = u_r
                    v = v_r
                    p = p_r

                else:

                    # sample point is star right state
                    d = d_r*(p_star/p_r + (gamma - 1.0)/(gamma + 1.0))/(p_star*(gamma - 1.0)/((gamma + 1.0)*p_r) + 1.0)
                    u = u_r + (u_star - un_r)*n[0]
                    v = v_r + (u_star - un_r)*n[1]
                    p = p_star

            else:

                # right rarefaction
                s_hr = un_r + c_r

                if(0.0 >= s_hr):

                    # sample point is right data state
                    d = d_r
                    u = u_r
                    v = v_r
                    p = p_r

                else:

                    # sound speed of the star state and sound speed
                    # of the tail of the rarefraction
                    c_star_r = c_r*pow(p_star/p_r, (gamma-1.0)/(2.0*gamma))
                    s_tr = u_star + c_star_r

                    if(0.0 <= s_tr):

                        # sample point is star left state
                        d = d_r*pow(p_star/p_r, 1.0/gamma)
                        u = u_r + (u_star - un_r)*n[0]
                        v = v_r + (u_star - un_r)*n[1]
                        p = p_star

                    else:

                        # sampled point is inside right fan
                        c = (2.0/(gamma + 1.0))*(c_r - 0.5*(gamma - 1.0)*un_r)
                        u_tmp = (2.0/(gamma + 1.0))*(-c_r + 0.5*(gamma-1.0)*un_r)

                        d = d_r*pow(c/c_r, 2.0/(gamma - 1.0))
                        u = u_l + (u_tmp - un_r)*n[0]
                        v = v_l + (u_tmp - un_r)*n[1]
                        p = p_r*pow(c/c_r, 2.0*gamma/(gamma - 1.0))

        # create the flux vector
        wx = w_face[0,i]
        wy = w_face[1,i]
        un = u*n[0] + v*n[1]

        fluxes[0,i] = d*un
        fluxes[1,i] = d*u*un + p*n[0]
        fluxes[2,i] = d*v*un + p*n[1]
        fluxes[3,i] = (0.5*d*(u**2 + v**2) + gamma*p/(gamma-1.0))*un

        # add the deboost term
        fluxes[3,i] += 0.5*(wx**2 + wy**2)*fluxes[0,i] + wx*fluxes[1,i] + wy*fluxes[2,i]
        fluxes[1,i] += wx*fluxes[0,i]
        fluxes[2,i] += wy*fluxes[0,i]

def hllc(double[:,::1] face_l, double[:,::1] face_r, double[:,::1] flux, double[:,::1] normal, double[:,::1] w, double gamma, int num_faces):

    cdef double wn
    cdef int i, j
    cdef double s_l, s_r, s_contact
    cdef double d_l, u_l, v_l, p_l
    cdef double d_r, u_r, v_r, p_r
    cdef double un_l, un_r

    cdef double[:] flux_state = np.zeros(4, dtype=np.float64)
    cdef double[:] u_state_l = np.zeros(4, dtype=np.float64)
    cdef double[:] q_l = np.zeros(4, dtype=np.float64)
    cdef double[:] u_star_l = np.zeros(4, dtype=np.float64)
    cdef double[:] u_state_r = np.zeros(4, dtype=np.float64)
    cdef double[:] q_r = np.zeros(4, dtype=np.float64)
    cdef double[:] u_star_r = np.zeros(4, dtype=np.float64)
    cdef double[:] waves  = np.zeros(3, dtype=np.float64)
    cdef double[:] n = np.zeros(2, dtype=np.float64)

    for i in range(num_faces):

        d_l = face_l[0,i]
        u_l = face_l[1,i]
        v_l = face_l[2,i]
        p_l = face_l[3,i]

        d_r = face_r[0,i]
        u_r = face_r[1,i]
        v_r = face_r[2,i]
        p_r = face_r[3,i]

        n[0] = normal[0,i]
        n[1] = normal[1,i]

        # project velocity onto face normal
        un_l = u_l*n[0] + v_l*n[1]
        un_r = u_r*n[0] + v_r*n[1]

        get_waves(d_l, un_l, p_l, d_r, un_r, p_r, gamma, waves)
        s_l= waves[0]; s_contact = waves[1]; s_r = waves[2]

        for j in range(4):

            q_l[j] = face_l[j,i]
            q_r[j] = face_r[j,i]

        # convert from primitive variables to conserative variables
        prim_to_cons(u_state_l, q_l, gamma)
        prim_to_cons(u_state_r, q_r, gamma)

        # velocity of face projected onto face normal
        wn = w[0,i]*n[0] + w[1,i]*n[1]

        # calculate interface flux - eq. 10.71
        if(wn <= s_l):

            # left state
            construct_flux(flux_state, q_l, n, un_l, gamma)

            for j in range(4):
                flux_state[j] -= wn*u_state_l[j]

        elif((s_l < wn) and (wn <= s_r)):

            # intermediate state
            if(wn <= s_contact):

                # left star state
                hllc_state(u_star_l, q_l, n, un_l, gamma, s_l, s_contact)
                construct_flux(flux_state, q_l, n, un_l, gamma)

                for j in range(4):
                    flux_state[j] += s_l*(u_star_l[j] - u_state_l[j]) - wn*u_star_l[j]

            else:

                # right star state
                hllc_state(u_star_r, q_r, n, un_r, gamma, s_r, s_contact)
                construct_flux(flux_state, q_r, n, un_r, gamma)

                for j in range(4):
                    flux_state[j] += s_r*(u_star_r[j] - u_state_r[j]) - wn*u_star_r[j]

        else:

            # it's a right state
            construct_flux(flux_state, q_r, n, un_r, gamma)

            for j in range(4):
                flux_state[j] -= wn*u_state_r[j]

        # store the fluxes
        for j in range(4):
            flux[j,i] = flux_state[j]

def hll(double[:,::1] face_l, double[:,::1] face_r, double[:,::1] flux, double[:,::1] normal, double[:,::1] w, double gamma, int num_faces):

    cdef double wn
    cdef int i, j
    cdef double s_l, s_r, s_contact
    cdef double d_l, u_l, v_l, p_l
    cdef double d_r, u_r, v_r, p_r
    cdef double un_l, un_r

    cdef double[:] flux_state = np.zeros(4, dtype=np.float64)
    cdef double[:] flux_l     = np.zeros(4, dtype=np.float64)
    cdef double[:] flux_r     = np.zeros(4, dtype=np.float64)

    cdef double[:] u_state_l = np.zeros(4, dtype=np.float64)
    cdef double[:] q_l       = np.zeros(4, dtype=np.float64)

    cdef double[:] u_state_r = np.zeros(4, dtype=np.float64)
    cdef double[:] q_r       = np.zeros(4, dtype=np.float64)

    cdef double[:] waves = np.zeros(3, dtype=np.float64)

    cdef double[:] n = np.zeros(2, dtype=np.float64)

    for i in range(num_faces):

        d_l = face_l[0,i]
        u_l = face_l[1,i]
        v_l = face_l[2,i]
        p_l = face_l[3,i]

        d_r = face_r[0,i]
        u_r = face_r[1,i]
        v_r = face_r[2,i]
        p_r = face_r[3,i]

        n[0] = normal[0,i]
        n[1] = normal[1,i]

        # project velocity onto face normal
        un_l = u_l*n[0] + v_l*n[1]
        un_r = u_r*n[0] + v_r*n[1]

        get_waves(d_l, un_l, p_l, d_r, un_r, p_r, gamma, waves)
        s_l = waves[0]; s_contact = waves[1]; s_r = waves[2]

        for j in range(4):

            q_l[j] = face_l[j,i]
            q_r[j] = face_r[j,i]

        # convert from primitive variables to conserative variables
        prim_to_cons(u_state_l, q_l, gamma)
        prim_to_cons(u_state_r, q_r, gamma)

        wn = w[0,i]*n[0] + w[1,i]*n[1]

        # calculate interface flux - eq. 10.71
        if(wn <= s_l):

            # left state
            construct_flux(flux_state, q_l, n, un_l, gamma)

            for j in range(4):
                flux_state[j] -= wn*u_state_l[j]

        # star state 
        elif((s_l < wn) and (wn <= s_r)):

            construct_flux(flux_l, q_l, n, un_l, gamma)
            construct_flux(flux_r, q_r, n, un_r, gamma)

            for j in range(4):
                flux_state[j]  = 0.0
                flux_state[j] += (s_r*flux_l[j] - s_l*flux_r[j] + s_l*s_r*(u_state_r[j] - u_state_l[j]))/(s_r - s_l)
                flux_state[j] -= wn*(s_r*u_state_r[j] - s_l*u_state_l[j] + flux_l[j] - flux_r[j])/(s_r - s_l)

        else:

            # it's a right state
            construct_flux(flux_state, q_r, n, un_r, gamma)

            for j in range(4):
                flux_state[j] -= wn*u_state_r[j]

        for j in range(4):
            flux[j,i] = flux_state[j]


cdef get_waves(double d_l, double u_l, double p_l, double d_r, double u_r, double p_r, double gamma, double[:] waves):

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

    c_l = max(sqrt(gamma*p_l/d_l), c_floor)
    c_r = max(sqrt(gamma*p_r/d_r), c_floor)

    d_avg = 0.5*(d_l + d_r)
    c_avg = 0.5*(c_l + c_r)

    # estimate p* - eq. 9.20
    p_star = max(0.0, 0.5*(p_l + p_r) + 0.5*(u_l - u_r)*d_avg*c_avg)

    p_min = min(p_l, p_r)
    p_max = max(p_l, p_r)

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

    waves[0] = s_l
    waves[1] = s_c
    waves[2] = s_r




cdef prim_to_cons(double[:] u_s, double[:] q_s, double gamma):

    u_s[0] = q_s[0]
    u_s[1] = q_s[0]*q_s[1]
    u_s[2] = q_s[0]*q_s[2]
    u_s[3] = 0.5*q_s[0]*(q_s[1]*q_s[1] + q_s[2]*q_s[2]) + q_s[3]/(gamma-1.0)



cdef hllc_state(double[:] u_str, double[:] q, double[:] n, double un, double gamma, double s_wave, double contact_wave):

    cdef double factor = q[0]*(s_wave - un)/(s_wave - contact_wave)

    # star state - eq. 10.39
    u_str[0] = factor
    u_str[1] = factor*(q[1] + (contact_wave - un)*n[0]) # value in right frame 
    u_str[2] = factor*(q[2] + (contact_wave - un)*n[1]) # value in right frame 
    u_str[3] = factor*(0.5*(q[1]*q[1] + q[2]*q[2]) + q[3]/(q[0]*(gamma - 1.0)) +\
            (contact_wave - un)*(contact_wave + q[3]/(q[0]*(s_wave - un))))

cdef construct_flux(double[:] flux_state, double[:] q, double[:] n, double un, double gamma):

    flux_state[0] = q[0]*un
    flux_state[1] = q[0]*q[1]*un + q[3]*n[0]
    flux_state[2] = q[0]*q[2]*un + q[3]*n[1]
    flux_state[3] = (0.5*q[0]*(q[1]*q[1] + q[2]*q[2]) + q[3]*gamma/(gamma - 1.0))*un

#def p_guess(double d_l, double u_l, double p_l, double d_r, double u_r, double p_r, double gamma):
#
#    cdef double c_l, c_r
#    cdef double ppv
#    cdef double gl, gr, p_0
#    cdef double TOL = 1.0E-6
#
#    c_l = sqrt(gamma*p_l/d_l)  # left sound speed
#    c_r = sqrt(gamma*p_r/d_r)  # right sound speed
#
#    # initial guess for pressure
#    ppv = 0.5*(p_l + p_r) - 0.125*(u_r - u_l)*(d_l + d_r)*(c_l + c_r)
#
#    if (ppv < 0.0):
#        ppv = 0.0
#
#    gl = sqrt((2.0/(d_l*(gamma+1)))/((gamma-1)*p_l/(gamma+1) + ppv))
#    gr = sqrt((2.0/(d_r*(gamma+1)))/((gamma-1)*p_r/(gamma+1) + ppv))
#
#    p_0 = (gl*p_l + gr*p_r - (u_r - u_l))/(gr + gl)
#
#    if (p_0 < 0.0):
#        p_0 = TOL
#
#    return p_0


cdef p_guess(double d_l, double u_l, double p_l, double d_r, double u_r, double p_r, double gamma):

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

cdef p_func(double d, double u, double p, double gamma, double p_old):

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


cdef p_func_deriv(double d, double u, double p, double gamma, double p_old):

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


cdef get_pstar(double d_l, double u_l, double p_l, double d_r, double u_r, double p_r, double gamma):

    cdef double p_old
    cdef double u_diff = u_r - u_l
    cdef int i = 0
    cdef double TOL = 1.0E-6
    cdef double change, p, f_r, f_l, df_r, df_l
    cdef int MAX_ITER = 1000

    p_old = p_guess(d_l, u_l, p_l, d_r, u_r, p_r, gamma)

    while (i < MAX_ITER):

        f_l  = p_func(d_l, u_l, p_l, gamma, p_old)
        f_r  = p_func(d_r, u_r, p_r, gamma, p_old)
        df_l = p_func_deriv(d_l, u_l, p_l, gamma, p_old)
        df_r = p_func_deriv(d_r, u_r, p_r, gamma, p_old)

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
