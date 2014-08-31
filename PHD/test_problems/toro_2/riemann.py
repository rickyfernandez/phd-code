import h5py
import numpy as np
from math import sqrt

def Derivative(pold, d_k, p_k, c_k, g):

    if (pold <= p_k):

        # Its a rarefraction wave
        p_ratio = pold/p_k
        f   = (2.0/(g-1.0)) *c_k *(p_ratio**((g-1.0)/(2.0*g)) - 1.0)
        f_d = (1.0/(d_k*c_k))*p_ratio**(-((g+1)/(2.0*g)))

    else:

        # Its a shock wave
        A_k = (2.0/(g+1.0))/d_k
        B_k = ((g-1.0)/(g+1.0))*p_k
        f   = (pold - p_k)*sqrt(A_k/(B_k + pold))
        f_d = (1.0 - 0.5*(pold - p_k)/(B_k + pold))*sqrt(A_k/(B_k + pold))


    return f, f_d

def State(pstar, ustar, s, dl, ul, pl, cl, dr, ur, pr, cr, g):

    if(s <= ustar):

        # sampling point lies to the left of the contact discontinuity
        if(pstar <= pl):

            # left rarefraction, below is the sound speed of the head
            s_HL = ul - cl

            if(s <= s_HL):

                # sample point is left data state
                d = dl
                u = ul
                p = pl

            else:

                # sound speed of the star state and sound speed
                # of the tail of the rarefraction
                c_star_l = cl*(pstar/pl)**((g-1.0)/(2.0*g))
                s_TL = ustar - c_star_l

                if(s >= s_TL):

                    # sample point is star left state
                    d = dl*(pstar/pl)**(1.0/g)
                    u = ustar
                    p = pstar

                else:

                    # sampled point is inside left fan
                    u = (2.0/(g+1.0))*(cl + 0.5*(g-1.0)*ul + s)
                    c = (2.0/(g+1.0))*(cl + 0.5*(g-1.0)*(ul - s))
                    d = dl*(c/cl)**(2.0/(g-1.0))
                    p = pl*(c/cl)**(2.0*g/(g-1.0))
        else:

            # left shock
            s_l = ul - cl*sqrt((g+1.0)*pstar/(2.0*g*pl) + (g-1.0)/(2*g))

            if(s <= s_l):

                # sampled point is a left data state
                d = dl
                u = ul
                p = pl

            else:

                # sampled point is star left state
                d = dl*((pstar/pl + (g-1.0)/(g+1.0))/(pstar*(g-1.0)/((g+1.0)*pl) + 1.0))
                u = ustar
                p = pstar


    else:

        # sampling point lies to the right of the contact
        if(pstar >= pr):

            # right shock
            s_r = ur + cr*sqrt((g+1.0)*pstar/(2.0*g*pr) + (g-1.0)/(2.0*g))

            if(s >= s_r):

                # sampled point is right data states
                d = dr
                u = ur
                p = pr

            else:

                # sample point is star right state
                d = dr*((pstar/pr + (g-1.0)/(g+1.0))/(pstar*(g-1.0)/((g+1.0)*pr) + 1.0))
                u = ustar
                p = pstar

        else:

            # right rarefaction
            s_HR = ur + cr

            if(s >= s_HR):

                # sample point is right data state
                d = dr
                u = ur
                p = pr

            else:

                # sound speed of the star state and sound speed
                # of the tail of the rarefraction
                c_star_r = cr*(pstar/pr)**((g-1.0)/(2.0*g))
                s_TR = ustar + c_star_r

                if(s <= s_TR):

                    # sample point is star left state
                    d = dr*(pstar/pr)**(1.0/g)
                    u = ustar
                    p = pstar

                else:

                    # sampled point is inside right fan
                    u = (2.0/(g+1.0))*(-cr + 0.5*(g-1.0)*ur + s)
                    c = (2.0/(g+1.0))*(cr - 0.5*(g-1.0)*(ur - s))
                    d = dr*(c/cr)**(2.0/(g-1.0))
                    p = pr*(c/cr)**(2.0*g/(g-1.0))

    return d, u, p

def riemann(domain_length=2.0, diaphram=1.0, cells=128, gamma=1.4, time_out=0.15,
        density_left=1.0, velocity_left=-2.0, pressure_left=0.4, density_right=1.0, velocity_right=2.0, pressure_right=0.4,
        mpa=1.0):

    # Exact Riemann solver for the Euler equations

    # compute the left and right sound speeds
    c_left  = sqrt(gamma*pressure_left/density_left)
    c_right = sqrt(gamma*pressure_right/density_right)

    # first we need an initial value for p* and u*
    # we begin with the PVRS scheme, see chapter 9.1
    Q_user = 2.0
    density_bar = 0.5*(density_left + density_right)
    c_bar       = 0.5*(c_left + c_right)

    pressure_initial  = 0.5*(pressure_left + pressure_right)
    pressure_initial += 0.5*(velocity_left - velocity_right)*density_bar*c_bar
    pressure_star     = max(0.0, pressure_initial)
    pressure_max      = max(pressure_left, pressure_right)
    pressure_min      = min(pressure_left, pressure_right)
    Q_max             = pressure_max/pressure_min

    if (Q_max <= Q_user and \
        pressure_min <= pressure_initial <= pressure_max):

        print " Initial guess for pressure: ", pressure_initial

    elif (pressure_initial <= pressure_min):

        P_LR   = (pressure_left/pressure_right)**((gamma - 1.0)/(2.0*gamma))
        u_star = (P_LR*velocity_left/c_left + velocity_right/c_right + 2*(P_LR - 1)/(gamma - 1))
        u_star = u_star/(P_LR/c_left + 1/c_right)
        P_TL   = (1 + (gamma -1)*(velocity_left - u_star)/(2*c_left))**(2.0*gamma/(gamma - 1.0))
        P_TR   = (1 + (gamma -1)*(u_star - velocity_right)/(2*c_right))**(2.0*gamma/(gamma - 1.0))

        pressure_initial = 0.5*(pressure_left*P_TL + pressure_right*P_TR)
        print "  Initial guess for pressure: ", pressure_initial

    else:

        g_L = sqrt((2.0/(gamma+1.0))/density_left)/(((gamma-1.0)/(gamma+1.0))*pressure_left + pressure_initial)
        g_R = sqrt((2.0/(gamma+1.0))/density_right)/(((gamma-1.0)/(gamma+1.0))*pressure_right + pressure_initial)

        pressure_initial = (g_L*pressure_left + g_R*pressure_right)/(g_L + g_R)
        print " Initial guess for pressure: ", pressure_initial


    # Now we begin Newton-Raphosn iteration to find p*
    print "-------------------------------------------------------------------"
    print "  Iteration      Delta                                             "
    print "-------------------------------------------------------------------"

    i = 1
    tolerance     = 1.0E-6
    pressure_old  = pressure_initial
    velocity_diff = velocity_right - velocity_left
    while(i <= 20):

        FL, FL_Derivative = Derivative(pressure_old, density_left, pressure_left, c_left, gamma)
        FR, FR_Derivative = Derivative(pressure_old, density_right, pressure_right, c_right, gamma)
        pressure_star = pressure_old - (FL + FR + velocity_diff)/(FL_Derivative + FR_Derivative)
        change = 2.0*abs((pressure_star - pressure_old)/(pressure_star + pressure_old))
        print "%3d\t%.5e " %(i, change)
        if (change <= tolerance):
            break
        if (pressure_star <= 0.0):
            pressure_star = tolerence
        pressure_old = pressure_star; i+=1

    # compute velocity in the star region
    velocity_star = 0.5*(velocity_left + velocity_right + FR - FL)

    print "-------------------------------------------------------------------"
    print "  Pressure       Velocity                                          "
    print "-------------------------------------------------------------------"
    print pressure_star/mpa, velocity_star
    print "\n"

    # Now beggin to plot the data
    x = []; d = []; u = []; p = [];
    dx = domain_length/cells

    for i in range(cells):
        x.append((i+0.5)*dx)

    for i in range(cells):
        if(x[i] <= diaphram):
            d.append(density_left)
            u.append(velocity_left)
            p.append(pressure_left)
        else:
            d.append(density_right)
            u.append(velocity_right)
            p.append(pressure_right)

    for i in range(cells):
        s = (x[i]-diaphram)/time_out

        d[i], u[i], p[i] = State(pressure_star, velocity_star, s,
                                 density_left,  velocity_left,  pressure_left,  c_left,
                                 density_right, velocity_right, pressure_right, c_right,
                                 gamma)

    f = h5py.File("riemann_sol.hdf5", "w")
    f["/x"] = np.array(x)
    f["/density"] = np.array(d)
    f["/velocity"] = np.array(u)
    f["/pressure"] = np.array(p)

if __name__ == "__main__":

    riemann()
