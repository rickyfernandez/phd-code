from libc.math cimport sqrt, pow, fabs
import numpy as np
cimport numpy as np
cimport cython


def hllc(double[:,::1] face_l, double[:,::1] face_r, double[:,::1] flux, double[:] w_face, double gamma, int num_faces):

    cdef double w
    cdef int i, j
    cdef double s_l, s_r, s_contact
    cdef double d_lef, p_l, u_l
    cdef double d_r, p_r, u_r

    cdef double[:] flux_state = np.zeros(4, dtype=np.float64)
    cdef double[:] u_state_l = np.zeros(4, dtype=np.float64)
    cdef double[:] q_l = np.zeros(4, dtype=np.float64)
    cdef double[:] u_star_l = np.zeros(4, dtype=np.float64)
    cdef double[:] u_state_r = np.zeros(4, dtype=np.float64)
    cdef double[:] q_r = np.zeros(4, dtype=np.float64)
    cdef double[:] u_star_r = np.zeros(4, dtype=np.float64)
    cdef double[:] waves  = np.zeros(3, dtype=np.float64)

    for i in range(num_faces):

        d_l = face_l[0,i]
        u_l = face_l[1,i]
        p_l = face_l[3,i]

        d_r = face_r[0,i]
        u_r = face_r[1,i]
        p_r = face_r[3,i]

        get_waves(d_l, u_l, p_l, d_r, u_r, p_r, gamma, waves)
        s_l= waves[0]; s_contact = waves[1]; s_r = waves[2]

        for j in range(4):

            q_l[j] = face_l[j,i]
            q_r[j] = face_r[j,i]

        # convert from primitive variables to conserative variables
        prim_to_cons(u_state_l,  q_l, gamma)
        prim_to_cons(u_state_r, q_r, gamma)

        w = w_face[i]

        # calculate interface flux - eq. 10.71
        if(w <= s_l):
            # left state
            construct_flux(flux_state, q_l, gamma)

            for j in range(4):
                flux_state[j] -= w*u_state_l[j]

        elif((s_l < w) and (w <= s_r)):

            if(w <= s_contact):

                hllc_state(u_star_l, q_l, gamma, s_l, s_contact)
                construct_flux(flux_state, q_l, gamma)

                for j in range(4):
                    flux_state[j] += s_l*(u_star_l[j] - u_state_l[j]) - w*u_star_l[j]

            else:

                hllc_state(u_star_r, q_r, gamma, s_r, s_contact)
                construct_flux(flux_state, q_r, gamma)

                for j in range(4):
                    flux_state[j] += s_r*(u_star_r[j] - u_state_r[j]) - w*u_star_r[j]

        else:

            # it's a right state
            construct_flux(flux_state, q_r, gamma)

            for j in range(4):
                flux_state[j] -= w*u_state_r[j]

        for j in range(4):
            flux[j,i] = flux_state[j]

def hll(double[:,::1] face_l, double[:,::1] face_r, double[:,::1] flux, double[:] w_face, double gamma, int num_faces):

    cdef double w
    cdef int i, j
    cdef double s_l, s_r, s_contact
    cdef double d_l, p_l, u_l
    cdef double d_r, p_r, u_r

    cdef double[:] flux_state = np.zeros(4, dtype=np.float64)
    cdef double[:] flux_l     = np.zeros(4, dtype=np.float64)
    cdef double[:] flux_r     = np.zeros(4, dtype=np.float64)

    cdef double[:] u_state_l = np.zeros(4, dtype=np.float64)
    cdef double[:] q_l       = np.zeros(4, dtype=np.float64)

    cdef double[:] u_state_r = np.zeros(4, dtype=np.float64)
    cdef double[:] q_r       = np.zeros(4, dtype=np.float64)

    cdef double[:] waves = np.zeros(3, dtype=np.float64)

    for i in range(num_faces):

        d_l = face_l[0,i]
        u_l = face_l[1,i]
        p_l = face_l[3,i]

        d_r = face_r[0,i]
        u_r = face_r[1,i]
        p_r = face_r[3,i]

        get_waves(d_l, u_l, p_l, d_r, u_r, p_r, gamma, waves)
        s_l = waves[0]; s_contact = waves[1]; s_r = waves[2]

        for j in range(4):

            q_l[j]  = face_l[j,i]
            q_r[j] = face_r[j,i]

        # convert from primitive variables to conserative variables
        prim_to_cons(u_state_l, q_l, gamma)
        prim_to_cons(u_state_r, q_r, gamma)

        w = w_face[i]

        # calculate interface flux - eq. 10.71
        if(w <= s_l):
            # l state
            construct_flux(flux_state, q_l, gamma)

            for j in range(4):
                flux_state[j] -= w*u_state_l[j]

        elif((s_l < w) and (w <= s_r)):

            construct_flux(flux_l, q_l, gamma)
            construct_flux(flux_r, q_r, gamma)

            for j in range(4):
                flux_state[j]  = 0.0
                flux_state[j] += (s_r*flux_l[j] - s_l*flux_r[j] + s_l*s_r*(u_state_r[j] - u_state_l[j]))/(s_r - s_l)
                flux_state[j] -= w*(s_r*u_state_r[j] - s_l*u_state_l[j] + flux_l[j] - flux_r[j])/(s_r - s_l)

        else:

            # it's a right state
            construct_flux(flux_state, q_r, gamma)

            for j in range(4):
                flux_state[j] -= w*u_state_r[j]

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


cdef hllc_state(double[:] u_str, double[:] q, double gamma, double s_wave, double contact_wave):

    cdef double factor = q[0]*(s_wave - q[1])/(s_wave - contact_wave)

    # star state - eq. 10.39
    u_str[0] = factor
    u_str[1] = factor*contact_wave
    u_str[2] = factor*q[2];
    u_str[3] = factor*(0.5*(q[1]*q[1] + q[2]*q[2]) + q[3]/(q[0]*(gamma - 1.0)) +\
            (contact_wave - q[1])*(contact_wave + q[3]/(q[0]*(s_wave - q[1]))))


cdef prim_to_cons(double[:] u_s, double[:] q_s, double gamma):

    u_s[0] = q_s[0]
    u_s[1] = q_s[0]*q_s[1]
    u_s[2] = q_s[0]*q_s[2]
    u_s[3] = 0.5*q_s[0]*(q_s[1]*q_s[1] + q_s[2]*q_s[2]) + q_s[3]/(gamma-1.0)


cdef construct_flux(double[:] flux_state, double[:] q, double gamma):

    flux_state[0] = q[0]*q[1]
    flux_state[1] = q[0]*q[1]*q[1] + q[3]
    flux_state[2] = q[0]*q[1]*q[2]
    flux_state[3] = q[1]*(0.5*q[0]*(q[1]*q[1] + q[2]*q[2]) + q[3]*gamma/(gamma - 1.0))


def castro(double[:,::1] face_left, double[:,::1] face_right, double[:,::1] face_state, double gamma, int num_faces):


    # Primative right and left states
    cdef double  rho_left, rho_right
    cdef double    u_left, u_right
    cdef double    v_left, v_right
    cdef double    p_left, p_right
    cdef double    e_left, e_right
    cdef double rhoe_left, rhoe_right

    # Primative star states
    cdef double p_star,         u_star
    cdef double rho_star_left,  rho_star_right
    cdef double rhoe_star_left, rhoe_star_right
    cdef double a_star_left,   a_star_right

    # Langrangian, sound, and shock speed
    cdef double C_left, C_right, a_left, a_right, s


    # The state at the orign of the interface
    cdef double rho_state
    cdef double u_state
    cdef double v_state
    cdef double p_state
    cdef double e_state
    cdef double a_state
    cdef double rhoe_state

    # Speed of head and tail of right and left rarefaction
    cdef double head_left,  tail_left, shock_left
    cdef double head_right, tail_right, shock_right

    cdef double alpha
    cdef double smallp   = 1.0E-10
    cdef double smalla   = 1.0E-10
    cdef double smallrho = 1.0E-10

#----------------------------------------------------------------------------------------
# Begin the calculation of the flux at the interface 


    # Loop through all interfaces
    cdef int i
    for i in range(num_faces):


        rho_left  = face_left[0,i]
        u_left    = face_left[1,i]
        v_left    = face_left[2,i]
        p_left    = face_left[3,i]
        rhoe_left = p_left/(gamma-1)


        rho_right  = face_right[0,i]
        u_right    = face_right[1,i]
        v_right    = face_right[2,i]
        p_right    = face_right[3,i]
        rhoe_right = p_right/(gamma-1)

        # Calculate left and right sound speed
        a_left  = max(sqrt(gamma*p_left/rho_left), smalla)
        a_right = max(sqrt(gamma*p_right/rho_right), smalla)


        # Calcualte the Lagrangian sound speed, eq. 9.25 (Toro)
        C_left  = max(sqrt(gamma*rho_left*p_left), smallrho*smalla)
        C_right = max(sqrt(gamma*rho_right*p_right), smallrho*smalla)


        # Linear approximation of star states, eq 9.28 (Toro)
        p_star = (C_right*p_left + C_left*p_right +
                  C_left*C_right*(u_left - u_right))/(C_left + C_right)
        p_star = max(p_star, smallp)

        u_star  = (C_left*u_left + C_right*u_right + p_left -
                  p_right)/(C_left + C_right)

        rho_star_left   = rho_left + (p_star - p_left)/(a_left*a_left)
        rho_star_right  = rho_right + (p_star - p_right)/(a_right*a_right)

        rhoe_star_left  = rhoe_left  + (p_star - p_left)*(rhoe_left/rho_left + p_left/rho_left)/(a_left*a_left)
        rhoe_star_right = rhoe_right + (p_star - p_right)*(rhoe_right/rho_right + p_right/rho_right)/(a_right*a_right)

        a_star_left  = max(sqrt(gamma*p_star/rho_star_left), smalla)
        a_star_right = max(sqrt(gamma*p_star/rho_star_right), smalla)


#----------------------------------------------------------------------------------------
# Now calculate what state the interface is in

       # Now we calculate which state we are in, see fig. 9.2 (Toro)
        # for a visual of possible wave patterns.
        if u_star > 0.0:

            v_state = v_left

            # The contact discontinuity is moving right so the origin
            # can be a left state or a left star state

            # Calculate the left head and tail wave speed
            head_left = u_left - a_left
            tail_left = u_star - a_star_left


            if p_star > p_left:

                # Its a shock, calculate the shock speed
                s = 0.5*(head_left + tail_left)

                if s > 0.0:

                    # shock is moving to the right, its a
                    # left state
                    rho_state  = rho_left
                    u_state    = u_left
                    p_state    = p_left
                    rhoe_state = rhoe_left

                else:

                    # shock is moving to the left, its a
                    # left star state
                    rho_state  = rho_star_left;
                    u_state    = u_star;
                    p_state    = p_star;
                    rhoe_state = rhoe_star_left;



            else:

                # The wave is rarefaction
                if (head_left < 0.0 and tail_left < 0.0):

                    # Rarefaction wave is moving to the left, its
                    # left star state
                    rho_state  = rho_star_left
                    u_state    = u_star
                    p_state    = p_star
                    rhoe_state = rhoe_star_left

                elif head_left > 0.0 and tail_left > 0.0:

                    # Rarefaction wave is moving to the right, its
                    # a left state
                    rho_state  = rho_left
                    u_state    = u_left
                    p_state    = p_left
                    rhoe_state = rhoe_left

                else:

                    # Rarefaction wave spans the origin, eq 35 (Almgren et al. 2010)
                    alpha = head_left/(head_left - tail_left)

                    rho_state  = alpha*rho_star_left  + (1.0 - alpha)*rho_left
                    u_state    = alpha*u_star         + (1.0 - alpha)*u_left
                    p_state    = alpha*p_star         + (1.0 - alpha)*p_left
                    rhoe_state = alpha*rhoe_star_left + (1.0 - alpha)*rhoe_left




        elif u_star < 0.0:


            v_state = v_right
            # The contact discontinuity is moving left so the origin
            # can be a right state or a right star state

            # Calculate the right head and tail wave speed
            head_right = u_right + a_right
            tail_right = u_star + a_star_right


            if p_star > p_right:

                # Its a shock, calculate the shock speed
                s = 0.5*(head_right + tail_right)

                if s > 0.0:

                    # shock is moving to the right, its a
                    # right star state
                    rho_state  = rho_star_right
                    u_state    = u_star
                    p_state    = p_star
                    rhoe_state = rhoe_star_right

                else:

                    # shock is moving to the left, its a
                    # right state
                    rho_state  = rho_right
                    u_state    = u_right
                    p_state    = p_right
                    rhoe_state = rhoe_right


            else:

                # The wave is rarefaction
                if head_right < 0.0 and tail_right < 0.0:

                    # rarefaction wave is moving to the left, its
                    # a right state
                    rho_state  = rho_right;
                    u_state    = u_right;
                    p_state    = p_right;
                    rhoe_state = rhoe_right;

                elif head_right > 0.0 and tail_right > 0.0:

                    # Rarefaction wave is moving to the right, its
                    # a right star state
                    rho_state  = rho_star_right
                    u_state    = u_star
                    p_state    = p_star
                    rhoe_state = rhoe_star_right

                else:

                    # Rarefaction wave spans the origin, eq 35 (Almgren et al. 2010)
                    alpha = head_right/(head_right - tail_right)

                    rho_state  = alpha*rho_star_right  + (1.0 - alpha)*rho_right
                    u_state    = alpha*u_star          + (1.0 - alpha)*u_right
                    p_state    = alpha*p_star          + (1.0 - alpha)*p_right
                    rhoe_state = alpha*rhoe_star_right + (1.0 - alpha)*rhoe_right


        else:

            rho_state  = 0.5*(rho_star_left + rho_star_right)
            u_state    = u_star
            v_state    = 0.5*(v_left + v_right)
            p_state    = p_star
            rhoe_state = 0.5*(rhoe_star_left + rhoe_star_right)

#----------------------------------------------------------------------------------------
# Calculate the flux at each interface */

        # Fluxes are left face valued for the i'th cell
        face_state[0,i] = rho_state
        face_state[1,i] = u_state
        face_state[2,i] = v_state
        face_state[3,i] = rhoe_state
        face_state[4,i] = p_state




def guessP(double rho_left, double u_left, double p_left, double rho_right, double u_right, double p_right, double gamma):

    cdef double c_left, c_right
    cdef double p_0
    cdef double gl, gr
    cdef double TOL = 1.0e-6

    cdef double Q_max, Q_user
    cdef double p_max, p_min
    cdef double P_LR, P_TR, P_TL
    cdef double u_star

    c_left  = sqrt(gamma*p_left/rho_left)     # left sound speed
    c_right = sqrt(gamma*p_right/rho_right)   # right sound speed

    # initial guess for pressure
    p_0 = 0.5*(p_left + p_right)-0.125*(u_right-u_left)*(rho_left+rho_right)*(c_left+c_right)

    p_max = max(p_left, p_right)
    p_min = min(p_left, p_right)
    Q_max = p_max/p_min

#--->
    Q_user = 2.0
    if ((Q_max <= Q_user) and (p_min <= p_0 <= p_max)):

        pass

    elif (p_0 <= p_min):

        P_LR   = pow((p_left/p_right),((gamma - 1.0)/(2.0*gamma)))
        u_star = (P_LR*u_left/c_left + u_right/c_right + 2*(P_LR - 1)/(gamma - 1))
        u_star = u_star/(P_LR/c_left + 1/c_right)
        P_TL   = pow((1 + (gamma -1)*(u_left - u_star)/(2*c_left)), (2.0*gamma/(gamma - 1.0)))
        P_TR   = pow((1 + (gamma -1)*(u_star - u_right)/(2*c_right)), (2.0*gamma/(gamma - 1.0)))

        p_0 = 0.5*(p_left*P_TL + p_right*P_TR)

    else:

        gl = sqrt((2.0/(rho_left*(gamma+1)))/((gamma-1)*p_left/(gamma+1) + p_0))
        gr = sqrt((2.0/(rho_right*(gamma+1)))/((gamma-1)*p_right/(gamma+1) + p_0))

        p_0 = (gl*p_left + gr*p_right - (u_right-u_left))/(gr + gl)


    #if (ppv < 0.0):
    #    ppv = 0.0

    #gl = sqrt((2.0/(rho_left*(gamma+1)))/((gamma-1)*p_left/(gamma+1) + ppv))
    #gr = sqrt((2.0/(rho_right*(gamma+1)))/((gamma-1)*p_right/(gamma+1) + ppv))

    #p_0 = (gl*p_left + gr*p_right - (u_right-u_left))/(gr + gl)

#--->

    if (p_0 < 0.0):
        p_0 = TOL

    return p_0


def PFunc(double rho, double u, double p, double gamma, double POld):

    cdef double f
    cdef double a = sqrt(gamma*p/rho)
    cdef double Ak, Bk

    # rarefaction wave
    if (POld <= p):
        f = 2*a/(gamma-1) * (pow(POld/p, (gamma-1)/(2*gamma)) - 1)

    # shock wave
    else:
        Ak = 2/(rho*(gamma+1))
        Bk = p*(gamma-1)/(gamma+1)

        f = (POld - p) * sqrt(Ak/(POld + Bk))

    return f


def PFuncDeriv(double rho, double u, double p, double gamma, double POld):

    cdef double fDer
    cdef double a = sqrt(gamma*p/rho)
    cdef double Ak, Bk

    # rarefaction wave
    if (POld <= p):
        fDer = 1/(a * rho) * pow(POld/p, -(gamma+1)/(2*gamma))

    # shock wave
    else:
        Ak = 2/(rho*(gamma+1))
        Bk = p*(gamma-1)/(gamma+1)

        fDer = sqrt(Ak/(POld + Bk))*(1.0 - 0.5*(POld - p)/(Bk + POld))

    return fDer


def get_pstar(double rho_left, double u_left, p_left, double rho_right, double u_right, double p_right, double gamma):

    cdef double POld
    cdef double VxDiff = u_right - u_left
    cdef int i = 0
    cdef double TOL = 1.0e-6
    cdef double change, p, fr, fl, frder, flder
    cdef int MAX_ITER = 100

    POld = guessP(rho_left, u_left, p_left, rho_right, u_right, p_right, gamma)

    while (i < MAX_ITER):

        fl = PFunc(rho_left, u_left, p_left, gamma, POld)
        fr = PFunc(rho_right, u_right, p_right, gamma, POld)
        flder = PFuncDeriv(rho_left, u_left, p_left, gamma, POld)
        frder = PFuncDeriv(rho_right, u_right, p_right, gamma, POld)

        p = POld - (fl + fr + VxDiff)/(flder + frder);

        change = 2.0*fabs((p-POld)/(p+POld))
        if (change <= TOL):
            return p

        if (p < 0.0):
            p = TOL

        POld = p
        i += 1


    # exit failure due to divergence


def exact(double[:,::1] face_left, double[:,::1] face_right, double[:,::1] face_state, double gamma, int num_faces):

    cdef double rho_left, u_left, v_left, p_left
    cdef double rho_right, u_right, v_right, p_right
    cdef double rho_state, u_state, v_state, p_state

    cdef double p_star, u_star
    cdef double fr, fl
    cdef double s_HL, s_TL, s_left, s_right, c
    cdef double c_left, c_right, c_star_left, c_star_right

    #if(!(Ul.d > 0.0)||!(Ur.d > 0.0))
    #ath_error("[exact flux]: Non-positive densities: dl = %e  dr = %e\n", 
	#      Ul.d, Ur.d);

    for i in range(num_faces):

        rho_left  = face_left[0,i]
        u_left    = face_left[1,i]
        v_left    = face_left[2,i]
        p_left    = face_left[3,i]

        rho_right = face_right[0,i]
        u_right   = face_right[1,i]
        v_right   = face_right[2,i]
        p_right   = face_right[3,i]

        c_left  = sqrt(gamma*p_left/rho_left)   # left sound speed
        c_right = sqrt(gamma*p_right/rho_right) # right sound speed

        # newton rhapson 
        p_star = get_pstar(rho_left, u_left, p_left, rho_right, u_right, p_right, gamma)

        # calculate the contact wave speed
        fr = PFunc(rho_right, u_right, p_right, gamma, p_star)
        fl = PFunc(rho_left, u_left, p_left, gamma, p_star)
        u_star = 0.5*(u_left + u_right) + 0.5*(fr - fl)


        if(0.0 <= u_star):

            # sampling point lies to the left of the contact discontinuity
            if(p_star <= p_left):

                # left rarefraction, below is the sound speed of the head
                s_HL = u_left - c_left

                if(0.0 <= s_HL):

                    # sample point is left data state
                    rho_state = rho_left
                    u_state   = u_left
                    v_state   = v_left
                    p_state   = p_left

                else:

                    # sound speed of the star state and sound speed
                    # of the tail of the rarefraction
                    #c_star_left = c_left*(p_star/p_left)**((g-1.0)/(2.0*g))
                    c_star_left = pow(c_left*(p_star/p_left),((gamma-1.0)/(2.0*gamma)))
                    s_TL = u_star - c_star_left

                    if(0.0 >= s_TL):

                        # sample point is star left state
                        #d = dl*(pstar/pl)**(1.0/g)
                        rho_state = pow(rho_left*(p_star/p_left), (1.0/gamma))
                        u_state   = u_star
                        v_state   = v_left
                        p_state   = p_star

                    else:

                        # sampled point is inside left fan
                        #c = (2.0/(gamma+1.0))*(c_left + 0.5*(gamma-1.0)*(u_left - s))
                        #u = (2.0/(g+1.0))*(cl + 0.5*(g-1.0)*ul + s)

                        c = (2.0/(gamma+1.0))*(c_left + 0.5*(gamma-1.0)*u_left)

                        rho_state = pow(rho_left*(c/c_left), (2.0/(gamma-1.0)))
                        u_state   = (2.0/(gamma+1.0))*(c_left + 0.5*(gamma-1.0)*u_left)
                        v_state   = v_left
                        p_state   = pow(p_left*(c/c_left), (2.0*gamma/(gamma-1.0)))
            else:

                # left shock
                s_left = u_left - c_left*sqrt((gamma+1.0)*p_star/(2.0*gamma*p_left) + (gamma-1.0)/(2.0*gamma))

                if(0.0 <= s_left):

                    # sampled point is a left data state
                    rho_state = rho_left
                    u_state   = u_left
                    v_state   = v_left
                    p_state   = p_left

                else:

                    # sampled point is star left state
                    rho_state = rho_left*((p_star/p_left + (gamma-1.0)/(gamma+1.0))/(p_star*(gamma-1.0)/((gamma+1.0)*p_left) + 1.0))
                    u_state   = u_star
                    v_state   = v_left
                    p_state   = p_star


        else:

            # sampling point lies to the right of the contact
            if(p_star >= p_right):

                # right shock
                s_right = u_right + c_right*sqrt((gamma+1.0)*p_star/(2.0*gamma*p_right) + (gamma-1.0)/(2.0*gamma))

                if(0.0 >= s_right):

                    # sampled point is right data states
                    rho_state = rho_right
                    u_state   = u_right
                    v_state   = v_right
                    p_state   = p_right

                else:

                    # sample point is star right state
                    rho_state = rho_right*((p_star/p_right + (gamma-1.0)/(gamma+1.0))/(p_star*(gamma-1.0)/((gamma+1.0)*p_right) + 1.0))
                    u_state   = u_star
                    v_state   = v_right
                    p_state   = p_star

            else:

                # right rarefaction
                s_HR = u_right + c_right

                if(0.0 >= s_HR):

                    # sample point is right data state
                    rho_state = rho_right
                    u_state   = u_right
                    v_state   = v_right
                    p_state   = p_right

                else:

                    # sound speed of the star state and sound speed
                    # of the tail of the rarefraction
                    c_star_right = pow(c_right*(p_star/p_right), ((gamma-1.0)/(2.0*gamma)))
                    s_TR = u_star + c_star_right

                    if(0.0 <= s_TR):

                        # sample point is star left state
                        rho_state = pow(rho_right*(p_star/p_right), (1.0/gamma))
                        u_state   = u_star
                        v_state   = v_right
                        p_state   = p_star

                    else:

                        # sampled point is inside right fan
                        c = (2.0/(gamma+1.0))*(c_right - 0.5*(gamma-1.0)*u_right)

                        rho_state = pow(rho_right*(c/c_right), (2.0/(gamma-1.0)))
                        u_state   = (2.0/(gamma+1.0))*(-c_right + 0.5*(gamma-1.0)*u_right)
                        v_state   = v_right
                        p_state   = pow(p_right*(c/c_right), (2.0*gamma/(gamma-1.0)))

        face_state[0,i] = rho_state
        face_state[1,i] = u_state
        face_state[2,i] = v_state
        face_state[3,i] = p_state/(gamma - 1.0)
        face_state[4,i] = p_state
