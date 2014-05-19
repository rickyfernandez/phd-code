import numpy as np

def hllc(ql, qr, gamma):

    floor_c = 1.0E-10 # put in global list

    #den_l = np.copy(ql[0,:]); den_r = np.copy(qr[0,:])
    #vel_l = np.copy(ql[1,:]); vel_r = np.copy(qr[1,:])
    #pre_l = np.copy(ql[3,:]); pre_r = np.copy(qr[3,:])

    den_l = ql[0,:]; den_r = qr[0,:]
    vel_l = ql[1,:]; vel_r = qr[1,:]
    pre_l = ql[3,:]; pre_r = qr[3,:]


    # calculate the sound speed
    c_l = np.maximum(np.sqrt(gamma*pre_l/den_l), floor_c)
    c_r = np.maximum(np.sqrt(gamma*pre_r/den_r), floor_c)

    den_avg = 0.5*(den_l + den_r)
    c_avg   = 0.5*(c_l + c_r)

    # Estimate p* - eq. 9.20
    pre_star = np.maximum(0.5*(pre_l + pre_r) + 0.5*(vel_l - vel_r)*den_avg*c_avg, 0.0)
    vel_star = np.zeros(den_l.shape)

    pre_min = np.minimum(pre_l, pre_r)
    pre_max = np.maximum(pre_l, pre_r)


    #-----------------------------------------------------------------------------
    # compute star estimate quantities by adaptive method
    #-----------------------------------------------------------------------------
    # if part
    i = (pre_max/pre_min < 2.0) & (pre_min < pre_star) & (pre_star < pre_max)
    if i.any():
        vel_star[i] = 0.5*(vel_l[i] + vel_r[i])\
            +0.5*(pre_l[i] - pre_r[i])/(den_avg[i]*c_avg[i])
    
    # else if part
    j = pre_star <= pre_min; j *= ~i
    if j.any():

        # Two rarefaction riemann solver (TRRS)
        # eq. 9.31
        z = (gamma - 1.0)/(2.0*gamma)

        # eq. 9.35
        p_lr = (pre_l[j]/pre_r[j])**z

        vel_star[j] = (p_lr*vel_l[j]/c_l[j] + vel_r[j]/c_r[j] + 2.0*(p_lr - 1.0)/(gamma - 1.0))\
            /(p_lr/c_l[j] + 1.0/c_r[j])

        # Estimate p* from two rarefaction aprroximation - 9.36
        pre_star[j]  = 0.5*pre_l[j]*(1.0 + (gamma - 1.0)*(vel_l[j] - vel_star[j])/(2.0*c_l[j]))**(1.0/z)
        pre_star[j] += 0.5*pre_r[j]*(1.0 + (gamma - 1.0)*(vel_star[j] - vel_r[j])/(2.0*c_r[j]))**(1.0/z)

    # else part
    k = ~(i+j)
    if k.any():

        # Two shock riemann solver (TSRS)
        # eq. 9.31
        A_l = 2.0/((gamma + 1.0)*den_l[k])
        A_r = 2.0/((gamma + 1.0)*den_r[k])

        B_l = pre_l[k]*((gamma - 1.0)/(gamma + 1.0))
        B_r = pre_r[k]*((gamma - 1.0)/(gamma + 1.0))

        # eq. 9.41
        g_l = np.sqrt(A_l/(pre_star[k] + B_l))
        g_r = np.sqrt(A_r/(pre_star[k] + B_r))

        # Estimate p* frow two shock aprroximation - 9.43
        pre_star[k] = (g_l*pre_l[k] + g_r*pre_r[k] - (vel_r[k] - vel_l[k]))/(g_l + g_r)
        vel_star[k] = 0.5*(vel_l[k] + vel_r[k]) + 0.5*(g_r*(pre_star[k] - pre_r[k])\
                -g_l*(pre_star[k] - pre_l[k]))


    #-----------------------------------------------------------------------------
    # calculate left, contact, right wave speeds 
    #-----------------------------------------------------------------------------
    # Calculate fastest left wave speed estimates - eq. 10.68-10.69
    s_l = np.zeros(ql.shape[1])
    s_r = np.zeros(ql.shape[1])

    # if part
    i = pre_star <= pre_l
    if i.any():
        # Rarefaction wave
        s_l[i] = vel_l[i] - c_l[i]

    # else part
    j = ~i
    if j.any():
        # Shock wave
        s_l[j] = vel_l[j] - c_l[j]*np.sqrt(1.0+((gamma+1.0)/(2.0*gamma))\
            *(pre_star[j]/pre_l[j] - 1.0))

    # Calculate fastest r wave speed estimates - 10.68-10.69
    # if part
    i = pre_star <= pre_r
    if i.any():
        # Rarefaction wave
        s_r[i] = vel_r[i] + c_r[i]

    # else part
    j = ~i
    if j.any():
        # Shock wave
        s_r[j] = vel_r[j] + c_r[j]*np.sqrt(1.0+((gamma+1.0)/(2.0*gamma))\
            *(pre_star[j]/pre_r[j] - 1.0))


#    # Contact wave speed - 10.70
#    i = slice(first, last+1)
#    s_contact = np.zeros(ql.shape[1])
#    s_contact[i] = (pre_r[i] - pre_l[i] + den_l[i]*vel_l[i]*(s_l[i] - vel_l[i])\
#            -den_r[i]*vel_r[i]*(s_r[i] - vel_r[i]))\
#            /(den_l[i]*(s_l[i] - vel_l[i]) - den_r[i]*(s_r[i] - vel_r[i]))

    # Contact wave speed - 10.70
    s_contact = np.zeros(ql.shape[1])
    s_contact = (pre_r - pre_l + den_l*vel_l*(s_l - vel_l)\
            -den_r*vel_r*(s_r - vel_r))/(den_l*(s_l - vel_l) - den_r*(s_r - vel_r))

    #-----------------------------------------------------------------------------
    # calculate flux at the interface 
    #-----------------------------------------------------------------------------
    # Convert from primitive variables to conserative variables
    ul = PrimToCons2D(ql, gamma)
    ur = PrimToCons2D(qr, gamma)

    flux = np.zeros(ql.shape)
    u_star_l = np.zeros(ql.shape)
    u_star_r = np.zeros(ql.shape)

    # Calculate interface flux - 10.71
    # if part
    i = 0.0 <= s_l
    if i.any():

        # left state 
        ConstructFlux2D(flux, ql, gamma, i)

    # else if part
    j = (s_l < 0.0) & (s_r >= 0.0)
    if j.any():

        k = s_contact >= 0.0; k *= j
        if k.any():

            HllcState2D(u_star_l, ql, gamma, s_l, s_contact, k)
            ConstructFlux2D(flux, ql, gamma, k)
            for f, ust, us in zip(flux, u_star_l, ul):
                f[k] += s_l[k]*(ust[k] - us[k])

        # else part
        l = ~k*j
        if l.any():

            HllcState2D(u_star_r, qr, gamma, s_r, s_contact, l)
            ConstructFlux2D(flux, qr, gamma, l)
            for f, ust, us in zip(flux, u_star_r, ur):
                f[l] += s_r[l]*(ust[l] - us[l])

    # else part
    m = ~(i+j)
    if m.any():

        # It's a right state
        ConstructFlux2D(flux, qr, gamma, m)

    return flux

def ConstructFlux2D(flux, q, gamma, i):

    # need to get rid of magic number and 2d condition
    flux[0,i] = q[0,i]*q[1,i]
    flux[1,i] = q[0,i]*q[1,i]**2 + q[3,i]
    flux[2,i] = q[0,i]*q[1,i]*q[2,i]
    flux[3,i] = q[1,i]*(0.5*q[0,i]*(q[1,i]**2 + q[2,i]**2) + q[3,i]*gamma/(gamma - 1.0))

def PrimToCons2D(q, gamma):
    
    # need to get rid of magic number and 2d condition
    u = np.zeros(q.shape)
    u[0,:] = q[0,:]
    u[1,:] = q[0,:]*q[1,:]
    u[2,:] = q[0,:]*q[2,:]
    u[3,:] = 0.5*q[0,:]*(q[1,:]**2 + q[2,:]**2) + q[3,:]/(gamma - 1.0)
    return u

def HllcState2D(u_str, q, gamma, sw, cw, i):

    factor = q[0,i]*((sw[i]-q[1,i])/(sw[i] - cw[i]))
    
    # Star state - eq. 10.39 (Toro 2009)
    u_str[0,i] = factor
    u_str[1,i] = factor*cw[i]
    u_str[2,i] = factor*q[2,i]
    u_str[3,i] = factor*(0.5*(q[1,i]** 2.0 + q[2,i]**2) + q[3,i]/(q[0,i]*(gamma - 1.0))\
            +(cw[i] - q[1,i])*(cw[i] + q[3][i]/(q[0,i]*(sw[i] - q[1,i]))))


if __name__ == "__main__":

    left  = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    right = np.array([[0.125, 0.125, 0.125], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])

    print Hllc(left, right, 1.4)
