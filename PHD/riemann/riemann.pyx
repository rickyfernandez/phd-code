import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double sqrt(double x)

#def CASTRO_c(np.ndarray VL,
#           np.ndarray VR,
#           np.ndarray F,
#           first_cell,
#           last_cell,
#           gamma):
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




#        rho_left  = VL[DENS][i]
#        u_left    = VL[VELX][i]
#        p_left    = VL[PRES][i]
#        rhoe_left = p_left/(gamma-1)
#
#
#        rho_right  = VR[DENS][i]
#        u_right    = VR[VELX][i]
#        p_right    = VR[PRES][i]
#        rhoe_right = p_right/(gamma-1)

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
