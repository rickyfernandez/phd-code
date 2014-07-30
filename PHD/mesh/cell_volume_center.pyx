from libc.math cimport sqrt, atan2
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cell_volume_center(double[:,::1] particles, int[:] neighbor_graph, int[:] num_neighbors,
        int[:] face_graph, double[:,::1] circum_centers, double[:] volume,
        double[:,::1] center_of_mass, int num_particles):

    cdef int id_p          # particle index 
    cdef int id_n          # particle index 
    cdef int ind           # neighbor index
    cdef int ind_face      # face vertex index

    cdef double xp, yp, x1, y1, x2, y2
    cdef double face_area, h

    cdef double[:] center_mass_face = np.zeros(2,dtype=np.float64)
    cdef double[:] center_of_mass_tri = np.zeros(2,dtype=np.float64)

    ind = 0
    ind_face = 0

    # loop over real particles
    for id_p in range(num_particles):

        # get poistion of particle
        xp = particles[0,id_p]
        yp = particles[1,id_p]

        # loop over neighbors
        for j in range(num_neighbors[id_p]):

            # index of neighbor
            id_n = neighbor_graph[ind]

            # neighbor position
            xn = particles[0,id_n]
            yn = particles[1,id_n]

            # distance between particles
            h = 0.5*sqrt((xn-xp)*(xn-xp) + (yn-yp)*(yn-yp))

            # calculate area of face between particle and neighbor 
            # in 2d each face is made up of two vertices
            x1 = circum_centers[face_graph[ind_face],0]
            y1 = circum_centers[face_graph[ind_face],1]
            ind_face += 1

            x2 = circum_centers[face_graph[ind_face],0]
            y2 = circum_centers[face_graph[ind_face],1]
            ind_face += 1

            face_area = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
            volume[id_p] += 0.5*face_area*h

            # center of mass of face
            center_mass_face[0] = 0.5*(x1 + x2)
            center_mass_face[1] = 0.5*(y1 + y2)

            # center of mass of triangle made from face as base and 
            # particle position as vertex
            center_of_mass_tri[0] = 2.0*center_mass_face[0]/3.0 + xp/3.0
            center_of_mass_tri[1] = 2.0*center_mass_face[1]/3.0 + yp/3.0

            # center mass of voronoi cell of particle id_p
            center_of_mass[0,id_p] += 0.5*face_area*h*center_of_mass_tri[0]
            center_of_mass[1,id_p] += 0.5*face_area*h*center_of_mass_tri[1]

            # go to next neighbor
            ind += 1

        center_of_mass[0,id_p] /= volume[id_p]
        center_of_mass[1,id_p] /= volume[id_p]


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
def faces_for_flux(double[:] face_areas, double[:,::1] face_velocities, double[:] face_angles, int[:,::1] face_pairs, double[:,::1] face_com,
        double[:,::1] particles, int[:] neighbor_graph, int[:] neighbor_graph_size, int[:] face_graph, double[:,::1] circum_centers,
        double[:,::1] w, int num_particles):

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
                #faces_info[1, k] = sqrt(x*x + y*y)
                face_areas[k] = sqrt(x*x + y*y)

                # step 2: calculate angle of normal
                xr = xn - xp
                yr = yn - yp

                # make sure the normal is pointing in the neighbor direction
                if (xr*y - yr*x) > 0.0:
                    x, y = y, -x
                else:
                    x, y = -y, x

                #faces_info[0, k] = atan2(y, x)
                face_angles[k] = atan2(y, x)

                # step 3: calculate velocity of face
                #faces_info[2, k] = 0.5*(w[0, id_p] + w[0, id_n])
                #faces_info[3, k] = 0.5*(w[1, id_p] + w[1, id_n])
                face_velocities[0,k] = 0.5*(w[0, id_p] + w[0, id_n])
                face_velocities[1,k] = 0.5*(w[1, id_p] + w[1, id_n])

                fx = 0.5*(x1 + x2)
                fy = 0.5*(y1 + y2)

                face_com[0,k] = fx
                face_com[1,k] = fy

                factor  = (w[0,id_p]-w[0,id_n])*(fx-0.5*(xp+xn)) + (w[1,id_p]-w[1,id_n])*(fy-0.5*(yp+yn))
                factor /= (xn-xp)*(xn-xp) + (yn-yp)*(yn-yp)

                #faces_info[2, k] += factor*(xn-xp)
                #faces_info[3, k] += factor*(yn-yp)
                face_velocities[0,k] += factor*(xn-xp)
                face_velocities[1,k] += factor*(yn-yp)

                # step 4: store the particles that make up the face
                #faces_info[4, k] = id_p
                #faces_info[5, k] = id_n
                face_pairs[0,k] = id_p
                face_pairs[1,k] = id_n

                # update counter
                k   += 1
                ind += 1

            else:

                # face accounted for, go to next neighbor and face
                ind += 1
                ind_face += 2
