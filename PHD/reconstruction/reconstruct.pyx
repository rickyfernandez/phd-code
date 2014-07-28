from libc.math cimport sqrt, atan2
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def number_of_faces(int[:] neighbor_graph, int[:] neighbor_graph_size, int num_particles):

    cdef int num_faces
    cdef int id_p, id_n
    cdef int ind, j

    ind = 0
    num_faces = 0

    # determine the number of faces 
    for id_p in range(num_particles):
        for j in range(neighbor_graph_size[id_p]):

            # index of neighbor
            id_n = neighbor_graph[ind]
            if id_n > id_p:
                num_faces += 1

            # go to next neighbor
            ind += 1

    return num_faces

@cython.boundscheck(False)
@cython.wraparound(False)
def faces_for_flux(double[:,::1] particles, int[:] neighbor_graph, int[:] neighbor_graph_size,
        int[:] face_graph, double[:,::1] circum_centers, double[:,::1] w, double[:,::1] faces_info, int num_particles):

    cdef int id_p, id_n, ind, ind_face, j, k
    cdef double xp, yp, xn, yn, x1, y1, x2, y2, xr, yr, x, y
    cdef double factor

    k = 0
    ind = 0
    ind_face = 0

    for id_p in range(num_particles):

        # position of particle
        xp = particles[0,id_p]
        yp = particles[1,id_p]

        for j in range(neighbor_graph_size[id_p]):

            # index of neighbor
            id_n = neighbor_graph[ind]

            if id_p < id_n:

                # step 1: calculate area of face, in 2d each face 
                # is made up of two vertices

                # position of neighbor
                xn = particles[0,id_n]
                yn = particles[1,id_n]

                # position of voronoi vertices that make up the face
                x1 = circum_centers[face_graph[ind_face],0]
                y1 = circum_centers[face_graph[ind_face],1]
                ind_face += 1

                x2 = circum_centers[face_graph[ind_face],0]
                y2 = circum_centers[face_graph[ind_face],1]
                ind_face += 1

                x = x2 - x1
                y = y2 - y1

                # area
                faces_info[1, k] = sqrt(x*x + y*y)

                # step 2: calculate angle of normal
                xr = xn - xp
                yr = yn - yp

                # make sure the normal is pointing in the neighbor direction
                if (xr*y - yr*x) > 0.0:
                    x, y = y, -x
                else:
                    x, y = -y, x

                faces_info[0, k] = atan2(y, x)

                # step 3: calculate velocity of face
                faces_info[2, k] = 0.5*(w[0, id_p] + w[0, id_n])
                faces_info[3, k] = 0.5*(w[1, id_p] + w[1, id_n])

                fx = 0.5*(x1 + x2)
                fy = 0.5*(y1 + y2)

                factor  = (w[0,id_p]-w[0,id_n])*(fx-0.5*(xp+xn)) + (w[1,id_p]-w[1,id_n])*(fy-0.5*(yp+yn))
                factor /= (xn-xp)*(xn-xp) + (yn-yp)*(yn-yp)

                faces_info[2, k] += factor*(xn-xp)
                faces_info[3, k] += factor*(yn-yp)

                # step 4: store the particles that make up the face
                faces_info[4, k] = id_p
                faces_info[5, k] = id_n

                # update counter
                k   += 1
                ind += 1

            else:

                # face accounted for, go to next neighbor and face
                ind += 1
                ind_face += 2

#def gradient(double[:,::1] particles, int[:] neighbor_graph, int[:] neighbor_graph_size,
#        int[:] face_graph, double[:,::1] circum_centers, double[:,::1] w, double[:,::1] faces_info, int num_real_particles):
#
#    cdef int id_p, id_n, ind
#    cdef int var, j
#    cdef double xp, yp
#
#    # loop over real particles
#    ind = 0
#    for id_p in range(num_real_particles):
#
#        # get particle position 
#        xp = particles[0,id_p]
#        yp = particles[1,id_p]
#
#        id_n = neighbor_graph[ind]
#        for var in range(4):
#
#            phi_max[var] = data[var,id_n]
#            phi_min[var] = data[var,id_n]
#
#            alpha[var] = 1.0
#
#        # loop over neighbors
#        for j in range(num_neighbors[id_p]):
#
#            # index of neighbor
#            id_n = neighbor_graph[ind]
#
#            # neighbor position
#            xn = particles[0,id_n]
#            yn = particles[1,id_n]
#
#            # calculate area of face between particle and neighbor 
#            # in 2d each face is made up of two vertices
#            x1 = circum_centers[face_graph[ind_face],0]
#            y1 = circum_centers[face_graph[ind_face],1]
#            ind_face += 1
#
#            x2 = circum_centers[face_graph[ind_face],0]
#            y2 = circum_centers[face_graph[ind_face],1]
#            ind_face += 1
#
#            face_area = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
#
#            fx = 0.5*(x1 + x2)
#            fy = 0.5*(y1 + y2)
#
#            cx = fx - 0.5*(xp + xn)
#            cy = fy - 0.5*(yp + yn)
#
#            rx = xp - xn
#            ry = yp - yn
#            r = sqrt(rx*rx + ry*ry)
#
#            for var in range(4):
#
#                phi_max[var] = max(phi_max[var], data[var,id_n])
#                phi_min[var] = min(phi_max[var], data[var,id_n])
#
#                gradx[var,id_p] += face_area*((data[var,id_n]-data[var,id_p])*cx - 0.5*(data[var,id_p] + data[var,id_n])*rx)/r
#                grady[var,id_p] += face_area*((data[var,id_n]-data[var,id_p])*cy - 0.5*(data[var,id_p] + data[var,id_n])*ry)/r
#
#            # go to next neighbor
#            ind += 1
#
#
#        for j in range(num_neighbors[id_p]):
#
#            # index of neighbor
#            id_n = neighbor_graph[ind2]
#
#            for var in range(4):
#
#                dphi = gradx[var,id_p]*(fx-sx) + grady[var,id_p]*(fy-sy)
#                if dphi > 0.0:
#                    psi = (phi_max[var] - data[var,id_p])/dphi
#                elif dphi < 0.0:
#                    psi = (phi_min[var] - data[var,id_p])/dphi
#                else:
#                    psi = 1.0
#
#                alpha[var] = min(alpha[var], psi)
#
#            ind2 += 1
#
#        for var in range(4):
#
#            gradx[var,id_p] *= alpha[var]
#            grady[var,id_p] *= alpha[var]
