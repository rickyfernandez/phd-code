from libc.math cimport sqrt, pow, fabs
import numpy as np
cimport numpy as np
cimport cython

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
