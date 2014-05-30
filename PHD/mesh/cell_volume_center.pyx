from libc.math cimport sqrt
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
        xp = particles[id_p,0]
        yp = particles[id_p,1]

        #print "particle in cython:", id_p
       
        # loop over neighbors
        for j in range(num_neighbors[id_p]):

            # index of neighbor
            id_n = neighbor_graph[ind]

            # neighbor position
            xn = particles[id_n,0]
            yn = particles[id_n,1]
            #print "neighbors in cython:", id_n

            # distance between particles
            h = 0.5*sqrt((xn-xp)*(xn-xp) + (yn-yp)*(yn-yp))
            #print "h in cython", h

            # calculate area of face between particle and neighbor 
            # in 2d each face is made up of two vertices
            x1 = circum_centers[face_graph[ind_face],0]
            y1 = circum_centers[face_graph[ind_face],1]
            ind_face += 1

            x2 = circum_centers[face_graph[ind_face],0]
            y2 = circum_centers[face_graph[ind_face],1]
            ind_face += 1

            face_area = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
            #print "volume in cython", 0.5*face_area*h
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
