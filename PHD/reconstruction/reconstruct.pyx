from libc.math cimport sqrt, atan2
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def gradient(double[:,::1] primitive, double[:,::1] gradx, double[:,::1] grady, double[:,::1] particles, double[:] volume,
        double[:,::1] center_of_mass, int[:] neighbor_graph, int[:] neighbor_graph_size, int[:] face_graph, double[:,::1] circum_centers,
        int num_real_particles):

    cdef int id_p, id_n, ind, ind2, ind_face, ind_face2
    cdef int var, j

    cdef double xp, yp, xn, yn, xf1, yf1, xf2, yf2
    cdef double fx, fy, cx, cy, rx, ry

    cdef double face_area, vol, r
    cdef double dph, psi

    cdef double[:] phi_max = np.zeros(4, dtype=np.float64)
    cdef double[:] phi_min = np.zeros(4, dtype=np.float64)
    cdef double[:] alpha   = np.zeros(4, dtype=np.float64)

    ind = ind2 = 0              # neighbor index for graph
    ind_face = ind_face2 = 0    # face index for graph

    # loop over real particles
    for id_p in range(num_real_particles):

        # particle position 
        xp = particles[0,id_p]
        yp = particles[1,id_p]

        # particle volume
        vol = volume[id_p]

        # set min/max to particle values 
        for var in range(4):

            phi_max[var] = primitive[var,id_p]
            phi_min[var] = primitive[var,id_p]
            alpha[var]   = 1.0

        # loop over neighbors of particle
        for j in range(neighbor_graph_size[id_p]):

            # index of neighbor
            id_n = neighbor_graph[ind]

            # neighbor position
            xn = particles[0,id_n]
            yn = particles[1,id_n]

            # coordinates that make up the face, in 2d a
            # face is made up of two points
            xf1 = circum_centers[face_graph[ind_face],0]
            yf1 = circum_centers[face_graph[ind_face],1]
            ind_face += 1

            xf2 = circum_centers[face_graph[ind_face],0]
            yf2 = circum_centers[face_graph[ind_face],1]
            ind_face += 1

            face_area = sqrt((xf2-xf1)*(xf2-xf1) + (yf2-yf1)*(yf2-yf1))

            # face center of mass
            fx = 0.5*(xf1 + xf2)
            fy = 0.5*(yf1 + yf2)

            # face center mass relative to midpoint of particles
            cx = fx - 0.5*(xp + xn)
            cy = fy - 0.5*(yp + yn)

            # separation vector of particles
            rx = xp - xn
            ry = yp - yn
            r = sqrt(rx*rx + ry*ry)

            for var in range(4):

                # add neighbor values to max and min
                phi_max[var] = max(phi_max[var], primitive[var,id_n])
                phi_min[var] = min(phi_min[var], primitive[var,id_n])

                # calculate gradient in each direction
                gradx[var,id_p] += face_area*((primitive[var,id_n]-primitive[var,id_p])*cx - 0.5*(primitive[var,id_p] + primitive[var,id_n])*rx)/(r*vol)
                grady[var,id_p] += face_area*((primitive[var,id_n]-primitive[var,id_p])*cy - 0.5*(primitive[var,id_p] + primitive[var,id_n])*ry)/(r*vol)

            # go to next neighbor
            ind += 1


        for j in range(neighbor_graph_size[id_p]):

            # index of neighbor
            id_n = neighbor_graph[ind2]

            xf1 = circum_centers[face_graph[ind_face2],0]
            yf1 = circum_centers[face_graph[ind_face2],1]
            ind_face2 += 1

            xf2 = circum_centers[face_graph[ind_face2],0]
            yf2 = circum_centers[face_graph[ind_face2],1]
            ind_face2 += 1

            fx = 0.5*(xf1 + xf2)
            fy = 0.5*(yf1 + yf2)

            for var in range(4):

                dphi = gradx[var,id_p]*(fx - center_of_mass[0,id_p]) + grady[var,id_p]*(fy - center_of_mass[1,id_p])
                if dphi > 0.0:
                    psi = (phi_max[var] - primitive[var,id_p])/dphi
                elif dphi < 0.0:
                    psi = (phi_min[var] - primitive[var,id_p])/dphi
                else:
                    psi = 1.0

                alpha[var] = min(alpha[var], psi)

            # go to next neighbor
            ind2 += 1

        for var in range(4):

            gradx[var,id_p] *= alpha[var]
            grady[var,id_p] *= alpha[var]



@cython.boundscheck(False)
@cython.wraparound(False)
def extrapolate(double[:,::1] left_face, double[:,::1] right_face, double[:,::1] gradx, double[:,::1] grady, double[:,::1] face_com,
        int[:,::1] face_pairs, double[:,::1] cell_com, double gamma, double dt, int num_faces):

    cdef int i, j, k, var
    cdef double rho_l, vx_l, vy_l, p_l
    cdef double rho_r, vx_r, vy_r, p_r

    for k in range(num_faces):

        i = face_pairs[0,k]
        j = face_pairs[1,k]

        # add temporal component
        rho_l = left_face[0,k]
        vx_l  = left_face[1,k]
        vy_l  = left_face[2,k]
        p_l   = left_face[3,k]

        rho_r = right_face[0,k]
        vx_r  = right_face[1,k]
        vy_r  = right_face[2,k]
        p_r   = right_face[3,k]

        # density
        #left_face[0,k]  -= 0.5*dt*(vx_l*gradx[0,i] + vy_l*grady[0,i] + rho_l*(gradx[1,i] + grady[2,i]))
        #right_face[0,k] -= 0.5*dt*(vx_r*gradx[0,j] + vy_r*grady[0,j] + rho_r*(gradx[1,j] + grady[2,j]))

        ## velocity x
        #left_face[1,k]  -= 0.5*dt*(vx_l*gradx[1,i] + vy_l*grady[1,i] + gradx[3,i]/rho_l)
        #right_face[1,k] -= 0.5*dt*(vx_r*gradx[1,j] + vy_r*grady[1,j] + gradx[3,j]/rho_r)

        ## velocity y
        #left_face[2,k]  -= 0.5*dt*(vx_l*gradx[2,i] + vy_l*grady[2,i] + grady[3,i]/rho_l)
        #right_face[2,k] -= 0.5*dt*(vx_r*gradx[2,j] + vy_r*grady[2,j] + grady[3,j]/rho_r)

        ## pressure
        #left_face[3,k]  -= 0.5*dt*(vx_l*gradx[3,i] + vy_l*grady[3,i] + gamma*p_l*(gradx[1,i] + grady[2,i]))
        #right_face[3,k] -= 0.5*dt*(vx_r*gradx[3,j] + vy_r*grady[3,j] + gamma*p_r*(gradx[1,j] + grady[2,j]))

        # add spatial component
        for var in range(4):

            left_face[var,k]  += gradx[var,i]*(face_com[0,k] - cell_com[0,i]) + grady[var,i]*(face_com[1,k] - cell_com[1,i])
            right_face[var,k] += gradx[var,j]*(face_com[0,k] - cell_com[0,j]) + grady[var,j]*(face_com[1,k] - cell_com[1,j])

            if (left_face[var,k] < 0.0) and (var == 0 or var == 3):
                print "left_face[", var, "],", k, "] = ", left_face[var,k]

            if (right_face[var,k] < 0.0) and (var == 0 or var == 3):
                print "right_face[", var, "],", k, "] = ", right_face[var,k]
