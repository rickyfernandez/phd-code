import numpy as np
from riemann_base import riemann_base

class pvrs(riemann_base):

    def flux(self, ql, qr, faces_info, gamma):

        # state variables in the face frame of reference
        left_face, right_face = self._transform_to_face(ql, qr, faces_info)


        # Begin the calculation of the flux at the interface 
        rho_left  = left_face[0,:]
        u_left    = left_face[1,:]
        v_left    = left_face[2,:]
        p_left    = left_face[3,:]
        rhoe_left = p_left/(gamma-1.)


        rho_right  = right_face[0,:]
        u_right    = right_face[1,:]
        v_right    = right_face[2,:]
        p_right    = right_face[3,:]
        rhoe_right = p_right/(gamma-1.)


        # Calculate left and right sound speed
        a_left  = np.maximum(np.sqrt(gamma*p_left/rho_left), self.small_sound_speed)
        a_right = np.maximum(np.sqrt(gamma*p_right/rho_right), self.small_sound_speed)


        # Calcualte the Lagrangian sound speed, eq. 9.25 (Toro)
        C_left  = np.maximum(np.sqrt(gamma*rho_left*p_left), self.small_rho*self.small_sound_speed)
        C_right = np.maximum(np.sqrt(gamma*rho_right*p_right), self.small_rho*self.small_sound_speed)


        # Linear approximation of star states, eq 9.28 (Toro)
        p_star = (C_right*p_left + C_left*p_right +
                  C_left*C_right*(u_left - u_right))/(C_left + C_right)
        p_star = np.maximum(p_star, self.small_pressure)

        u_star  = (C_left*u_left + C_right*u_right + p_left -
                  p_right)/(C_left + C_right)

        rho_star_left   = rho_left + (p_star - p_left)/(a_left*a_left)
        rho_star_right  = rho_right + (p_star - p_right)/(a_right*a_right)

        rhoe_star_left  = rhoe_left  + (p_star - p_left)*(rhoe_left/rho_left + p_left/rho_left)/(a_left*a_left)
        rhoe_star_right = rhoe_right + (p_star - p_right)*(rhoe_right/rho_right + p_right/rho_right)/(a_right*a_right)

        a_star_left  = np.maximum(np.sqrt(gamma*p_star/rho_star_left), self.small_sound_speed)
        a_star_right = np.maximum(np.sqrt(gamma*p_star/rho_star_right), self.small_sound_speed)


        #----------------------------------------------------------------------------------------
        # Now calculate what state the interface is in
        
        # Now we calculate which state we are in, see fig. 9.2 (Toro)
        # for a visual of possible wave patterns.

        head_left = np.zeros(rho_left.shape)
        tail_left = np.zeros(rho_left.shape)
        head_right = np.zeros(rho_left.shape)
        tail_right = np.zeros(rho_left.shape)
        s         = np.zeros(rho_left.shape)
        
        alpha = np.zeros(rho_left.shape)

        rho_state  = np.zeros(rho_left.shape)
        u_state    = np.zeros(rho_left.shape)
        v_state    = np.zeros(rho_left.shape)
        p_state    = np.zeros(rho_left.shape)
        rhoe_state = np.zeros(rho_left.shape)

        #if u_star > 0.0:
        i = u_star > 0.0
        if i.any():

            v_state[i] = v_left[i]

            # The contact discontinuity is moving right so the origin
            # can be a left state or a left star state

            # Calculate the left head and tail wave speed
            head_left[i] = u_left[i] - a_left[i]
            tail_left[i] = u_star[i] - a_star_left[i]


            #if p_star > p_left:
            j = p_star > p_left; ji=j*i
            if ji.any():

                # Its a shock, calculate the shock speed
                s[ji] = 0.5*(head_left[ji] + tail_left[ji])

                #if s > 0.0:
                k = s > 0.0; kji=k*ji
                if kji.any():

                    # shock is moving to the right, its a
                    # left state
                    rho_state[kji]  = rho_left[kji]
                    u_state[kji]    = u_left[kji]
                    p_state[kji]    = p_left[kji]
                    rhoe_state[kji] = rhoe_left[kji]

                #else:
                #l = ~k
                l = ~k*ji
                if l.any():
                     
                    # shock is moving to the left, its a
                    # left star state
                    rho_state[l]  = rho_star_left[l]
                    u_state[l]    = u_star[l]
                    p_state[l]    = p_star[l]
                    rhoe_state[l] = rhoe_star_left[l]


            #else:
            jj=~j*i
            if jj.any():

                # The wave is rarefaction
                #if (head_left < 0.0 and tail_left < 0.0):
                m = (head_left < 0.0) & (tail_left < 0.0); mjj=m*jj
                if mjj.any():

                    # Rarefaction wave is moving to the left, its
                    # left star state
                    rho_state[mjj]  = rho_star_left[mjj]
                    u_state[mjj]    = u_star[mjj]
                    p_state[mjj]    = p_star[mjj]
                    rhoe_state[mjj] = rhoe_star_left[mjj]

                #elif head_left > 0.0 and tail_left > 0.0:
                n = (head_left > 0.0) & (tail_left > 0.0); nmjj=~m*n*jj
                if nmjj.any():

                    # Rarefaction wave is moving to the right, its
                    # a left state
                    rho_state[nmjj]  = rho_left[nmjj]
                    u_state[nmjj]    = u_left[nmjj]
                    p_state[nmjj]    = p_left[nmjj]
                    rhoe_state[nmjj] = rhoe_left[nmjj]

                #else:
                mn = ~(m+n)*jj
                if mn.any():

                    # Rarefaction wave spans the origin, eq 35 (Almgren et al. 2010)
                    alpha[mn] = head_left[mn]/(head_left[mn] - tail_left[mn])

                    rho_state[mn]  = alpha[mn]*rho_star_left[mn]  + (1.0 - alpha[mn])*rho_left[mn]
                    u_state[mn]    = alpha[mn]*u_star[mn]         + (1.0 - alpha[mn])*u_left[mn]
                    p_state[mn]    = alpha[mn]*p_star[mn]         + (1.0 - alpha[mn])*p_left[mn]
                    rhoe_state[mn] = alpha[mn]*rhoe_star_left[mn] + (1.0 - alpha[mn])*rhoe_left[mn]

                

            
        #elif u_star < 0.0:
        r = u_star < 0.0; ri=~i*r
        if ri.any():


            v_state[ri] = v_right[ri]

            # The contact discontinuity is moving left so the origin
            # can be a right state or a right star state

            # Calculate the right head and tail wave speed
            head_right[ri] = u_right[ri] + a_right[ri]
            tail_right[ri] = u_star[ri] + a_star_right[ri]


            #if p_star > p_right:
            t = p_star > p_right; tri=t*ri
            if tri.any():

                # Its a shock, calculate the shock speed
                s[tri] = 0.5*(head_right[tri] + tail_right[tri])

                #if s > 0.0:
                q = s > 0.0; qtri=q*tri
                if qtri.any():


                    # shock is moving to the right, its a
                    # right star state
                    rho_state[qtri]  = rho_star_right[qtri]
                    u_state[qtri]    = u_star[qtri]
                    p_state[qtri]    = p_star[qtri]
                    rhoe_state[qtri] = rhoe_star_right[qtri]

                #else:
                qq = ~q*tri
                if qq.any():
                     
                    # shock is moving to the left, its a
                    # right state
                    rho_state[qq]  = rho_right[qq]
                    u_state[qq]    = u_right[qq]
                    p_state[qq]    = p_right[qq]
                    rhoe_state[qq] = rhoe_right[qq]


            #else:
            rr = ~t*ri
            if rr.any():

                # The wave is rarefaction
                #if head_right < 0.0 and tail_right < 0.0:
                h = (head_right < 0.0) & (tail_right < 0.0); hrr=h*rr
                if hrr.any():

                    # rarefaction wave is moving to the left, its
                    # a right state
                    rho_state[hrr]  = rho_right[hrr]
                    u_state[hrr]    = u_right[hrr]
                    p_state[hrr]    = p_right[hrr]
                    rhoe_state[hrr] = rhoe_right[hrr]

                #elif head_right > 0.0 and tail_right > 0.0:
                z = (head_right > 0.0) & (tail_right > 0.0); zrr =~h*z*rr
                if zrr.any():

                    # Rarefaction wave is moving to the right, its
                    # a right star state
                    rho_state[zrr]  = rho_star_right[zrr]
                    u_state[zrr]    = u_star[zrr]
                    p_state[zrr]    = p_star[zrr]
                    rhoe_state[zrr] = rhoe_star_right[zrr]

                #else:
                zt = ~(z+h)*rr
                if zt.any():

                    # Rarefaction wave spans the origin, eq 35 (Almgren et al. 2010)
                    alpha[zt] = head_right[zt]/(head_right[zt] - tail_right[zt])

                    rho_state[zt]  = alpha[zt]*rho_star_right[zt]  + (1.0 - alpha[zt])*rho_right[zt]
                    u_state[zt]    = alpha[zt]*u_star[zt]          + (1.0 - alpha[zt])*u_right[zt]
                    p_state[zt]    = alpha[zt]*p_star[zt]          + (1.0 - alpha[zt])*p_right[zt]
                    rhoe_state[zt] = alpha[zt]*rhoe_star_right[zt] + (1.0 - alpha[zt])*rhoe_right[zt]

                

        #else:
        iii = ~(i+r)
        if iii.any():

            rho_state[iii]  = 0.5*(rho_star_left[iii] + rho_star_right[iii])
            u_state[iii]    = u_star[iii]
            v_state[iii]    = 0.5*(u_left[iii] + v_left[iii])
            p_state[iii]    = p_star[iii]
            rhoe_state[iii] = 0.5*(rhoe_star_left[iii] + rhoe_star_right[iii])


        # calculate the flux at each face
        return self._transform_flux_to_lab(rho_state, u_state, v_state, rhoe_state, p_state, faces_info)



if __name__ == "__main__":

    left  = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    right = np.array([[0.125, 0.125, 0.125], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])

    print Hllc(left, right, 1.4)
