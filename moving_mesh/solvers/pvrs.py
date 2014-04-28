import numpy as np

def pvrs(ql, qr, faces_info, gamma):

    smallp   = 1.0E-10
    smalla   = 1.0E-10
    smallrho = 1.0E-10

    #----------------------------------------------------------------------------------------
    # Begin the calculation of the flux at the interface 
    theta = faces_info[0,:]

    # velocity of faces
    wx = faces_info[2,:]; wy = faces_info[3,:]

    # prepare faces to rotate and boost to face frame
    left_face = ql.copy(); right_face = qr.copy()

    # boost to frame of face
    left_face[1,:] -= wx; right_face[1,:] -= wx
    left_face[2,:] -= wy; right_face[2,:] -= wy

    # rotate to frame face
    u_left_rotated =  np.cos(theta)*left_face[1,:] + np.sin(theta)*left_face[2,:]
    v_left_rotated = -np.sin(theta)*left_face[1,:] + np.cos(theta)*left_face[2,:]

    left_face[1,:] = u_left_rotated
    left_face[2,:] = v_left_rotated

    u_right_rotated =  np.cos(theta)*right_face[1,:] + np.sin(theta)*right_face[2,:]
    v_right_rotated = -np.sin(theta)*right_face[1,:] + np.cos(theta)*right_face[2,:]

    right_face[1,:] = u_right_rotated
    right_face[2,:] = v_right_rotated

    #----------------------------------------------------------------------------------------
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
    a_left  = np.maximum(np.sqrt(gamma*p_left/rho_left), smalla)
    a_right = np.maximum(np.sqrt(gamma*p_right/rho_right), smalla)


    # Calcualte the Lagrangian sound speed, eq. 9.25 (Toro)
    C_left  = np.maximum(np.sqrt(gamma*rho_left*p_left), smallrho*smalla)
    C_right = np.maximum(np.sqrt(gamma*rho_right*p_right), smallrho*smalla)


    # Linear approximation of star states, eq 9.28 (Toro)
    p_star = (C_right*p_left + C_left*p_right +
              C_left*C_right*(u_left - u_right))/(C_left + C_right)
    p_star = np.maximum(p_star, smallp)

    u_star  = (C_left*u_left + C_right*u_right + p_left -
              p_right)/(C_left + C_right)

    rho_star_left   = rho_left + (p_star - p_left)/(a_left*a_left)
    rho_star_right  = rho_right + (p_star - p_right)/(a_right*a_right)

    rhoe_star_left  = rhoe_left  + (p_star - p_left)*(rhoe_left/rho_left + p_left/rho_left)/(a_left*a_left)
    rhoe_star_right = rhoe_right + (p_star - p_right)*(rhoe_right/rho_right + p_right/rho_right)/(a_right*a_right)

    a_star_left  = np.maximum(np.sqrt(gamma*p_star/rho_star_left), smalla)
    a_star_right = np.maximum(np.sqrt(gamma*p_star/rho_star_right), smalla)


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


    #----------------------------------------------------------------------------------------
    # Calculate the flux at each interface */

    F = np.zeros(ql.shape)
    G = np.zeros(ql.shape)

    # rotate state back to labrotary frame
    u = np.cos(theta)*u_state - np.sin(theta)*v_state
    v = np.sin(theta)*u_state + np.cos(theta)*v_state

    # unboost
    u_state = u + wx
    v_state = v + wy

    # calculate energy density in lab frame
    E_state = 0.5*rho_state*(u_state**2 + v_state**2) + rhoe_state

    F[0,:] = rho_state*(u_state - wx)
    F[1,:] = rho_state*u_state*(u_state-wx) + p_state
    F[2,:] = rho_state*v_state*(u_state-wx)
    F[3,:] = E_state*(u_state-wx) + p_state*u_state

    G[0,:] = rho_state*(v_state - wy)
    G[1,:] = rho_state*u_state*(v_state-wy)
    G[2,:] = rho_state*v_state*(v_state-wy) + p_state
    G[3,:] = E_state*(v_state-wy) + p_state*v_state

    # dot product flux in orientation of face
    return np.cos(theta)*F + np.sin(theta)*G



if __name__ == "__main__":

    left  = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    right = np.array([[0.125, 0.125, 0.125], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])

    print Hllc(left, right, 1.4)
