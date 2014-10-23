from libc.math cimport sqrt, atan2, sin, cos
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cell_face_info_2d(double[:,::1] particles, int[:] neighbor_graph, int[:] num_neighbors, int[:] face_graph, double[:,::1] circum_centers,
        double[:] volume, double[:,::1] center_of_mass,
        double[:] face_areas, double[:,::1] face_normal, int[:,::1] face_pairs, double[:,::1] face_com,
        int num_particles):

    cdef int id_p          # particle index 
    cdef int id_n          # neighbor index 
    cdef int ind           # loop index
    cdef int ind_face      # face vertex index

    cdef int j, k

    cdef double xp, yp, xn, yn, x1, y1, x2, y2, xr, yr, x, y
    cdef double face_area, h
    cdef double theta

    cdef double fx, fy, tx, ty

    k = 0
    ind = 0
    ind_face = 0

    # loop over real particles
    for id_p in range(num_particles):

        # get poistion of particle
        xp = particles[0,id_p]
        yp = particles[1,id_p]

        # loop over its neighbors
        for j in range(num_neighbors[id_p]):

            # index of neighbor
            id_n = neighbor_graph[ind]

            # neighbor position
            xn = particles[0,id_n]
            yn = particles[1,id_n]

            # half distance between particles
            h = 0.5*sqrt((xn-xp)*(xn-xp) + (yn-yp)*(yn-yp))

            # calculate area of face between particle and neighbor 
            # in 2d each face is made up of two vertices
            x1 = circum_centers[face_graph[ind_face],0]
            y1 = circum_centers[face_graph[ind_face],1]
            ind_face += 1 # go to next face vertex

            x2 = circum_centers[face_graph[ind_face],0]
            y2 = circum_centers[face_graph[ind_face],1]
            ind_face += 1 # go to next face

            x = x2 - x1
            y = y2 - y1

            # face area in 2d is length between vertices  
            face_area = sqrt(x*x + y*y)

            # the volume of the cell is the sum of triangle
            # areas - eq. 27
            volume[id_p] += 0.5*face_area*h

            # center of mass of face
            fx = 0.5*(x1 + x2)
            fy = 0.5*(y1 + y2)

            # center of mass of a triangle - eq. 31
            tx = 2.0*fx/3.0 + xp/3.0
            ty = 2.0*fy/3.0 + yp/3.0

            # center of mass of the cell is the sum weighted center of mass of
            # the triangles - eq. 29
            center_of_mass[0,id_p] += 0.5*face_area*h*tx
            center_of_mass[1,id_p] += 0.5*face_area*h*ty

            # store face information
            if id_p < id_n:

                # store the area of the face
                face_areas[k] = face_area

                # difference vector between particles 
                xr = xn - xp
                yr = yn - yp

                # make sure the normal is pointing toward the neighbor 
                if (xr*y - yr*x) > 0.0:
                    x, y = y, -x
                else:
                    x, y = -y, x

                # store the orientation of the norm of the face
                theta = atan2(y, x)
                face_normal[0,k] = cos(theta)
                face_normal[1,k] = sin(theta)

                # store the center mass of the face
                face_com[0,k] = fx
                face_com[1,k] = fy

                # store the particles that make up the face
                face_pairs[0,k] = id_p
                face_pairs[1,k] = id_n

                # go to next face 
                k += 1

            # go to next neighbor
            ind += 1

        # compelte the weighted sum for the cell - eq. 29
        center_of_mass[0,id_p] /= volume[id_p]
        center_of_mass[1,id_p] /= volume[id_p]


@cython.boundscheck(False)
@cython.wraparound(False)
def assign_face_velocities_2d(double[:,::1] particles, int[:] neighbor_graph, int[:] num_neighbors,
        double[:,::1] face_com, double[:,::1] face_velocities, double[:,::1] w, int num_particles):

    cdef int id_p, id_n, ind, j, k
    cdef double xp, yp, xn, yn
    cdef double factor

    k = 0
    ind = 0

    for id_p in range(num_particles):

        # position of particle
        xp = particles[0,id_p]
        yp = particles[1,id_p]

        for j in range(num_neighbors[id_p]):

            # index of neighbor
            id_n = neighbor_graph[ind]

            if id_p < id_n:

                # position of neighbor
                xn = particles[0,id_n]
                yn = particles[1,id_n]

                # the face velocity is approx the mean of velocities
                # of the particles 
                face_velocities[0,k] = 0.5*(w[0, id_p] + w[0, id_n])
                face_velocities[1,k] = 0.5*(w[1, id_p] + w[1, id_n])

                fx = face_com[0,k]
                fy = face_com[1,k]

                # correct face velocity due to residual motion - eq. 33
                factor  = (w[0,id_p]-w[0,id_n])*(fx-0.5*(xp+xn)) + (w[1,id_p]-w[1,id_n])*(fy-0.5*(yp+yn))
                factor /= (xn-xp)*(xn-xp) + (yn-yp)*(yn-yp)

                face_velocities[0,k] += factor*(xn-xp)
                face_velocities[1,k] += factor*(yn-yp)

                # update counter
                k   += 1
                ind += 1

            else:

                # face accounted for, go to next neighbor and face
                ind += 1


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

def assign_face_velocities_3d(double[:,::1] particles, int[:] neighbor_graph, int[:] num_neighbors,
        double[:,::1] face_com, double[:,::1] face_velocities, double[:,::1] w, int num_particles):

    cdef int id_p, id_n, ind, j, k
    cdef double xp, yp, zp, xn, yn, zn
    cdef double fx, fy, fz
    cdef double factor

    k = 0
    ind = 0

    for id_p in range(num_particles):

        # position of particle
        xp = particles[0,id_p]
        yp = particles[1,id_p]
        zp = particles[1,id_p]

        for j in range(num_neighbors[id_p]):

            # index of neighbor
            id_n = neighbor_graph[ind]

            if id_p < id_n:

                # position of neighbor
                xn = particles[0,id_n]
                yn = particles[1,id_n]
                zn = particles[2,id_n]

                # the face velocity is approx the mean of velocities
                # of the particles 
                face_velocities[0,k] = 0.5*(w[0, id_p] + w[0, id_n])
                face_velocities[1,k] = 0.5*(w[1, id_p] + w[1, id_n])
                face_velocities[2,k] = 0.5*(w[2, id_p] + w[2, id_n])

                fx = face_com[0,k]
                fy = face_com[1,k]
                fz = face_com[2,k]

                # correct face velocity due to residual motion - eq. 33
                factor  = (w[0,id_p]-w[0,id_n])*(fx-0.5*(xp+xn)) + (w[1,id_p]-w[1,id_n])*(fy-0.5*(yp+yn)) + (w[2,id_p]-w[2,id_n])*(fz-0.5*(zp+zn))
                factor /= (xn-xp)*(xn-xp) + (yn-yp)*(yn-yp) + (zn-zp)*(zn-zp)

                face_velocities[0,k] += factor*(xn-xp)
                face_velocities[1,k] += factor*(yn-yp)
                face_velocities[2,k] += factor*(zn-zp)

                # update counter
                k   += 1
                ind += 1

            else:

                # face accounted for, go to next neighbor and face
                ind += 1


def triangle_area(double[:,::1] t):
    """
    Purpose:
        compute the area of a triangle 3d

    Discussion:
        The area of a parallelogram is the cross product of two sides of the parallelogram.
        The triangle formed from the parallelogram is half the area

    Parameters:
        Input, 3x3 array of vertices of the triangle. Each column is a point and the rows
        are the x, y, and z dimension.

        Output, the area of the triangle.

    Author:
        Ricardo Fernandez

    Reference:
        www.people.sc.fsu.edu/~jburkardt/c_src/geometry/geometry.c
    """

    cdef double x, y, z

    # perform cross product from the vertices that make up the triangle
    # (v1 - v0) x (v2 - v0)
    x = (t[1,1] - t[1,0])*(t[2,2] - t[2,0]) - (t[2,1] - t[2,0])*(t[1,2] - t[1,0])
    y = (t[2,1] - t[2,0])*(t[0,2] - t[0,0]) - (t[0,1] - t[0,0])*(t[2,2] - t[2,0])
    z = (t[0,1] - t[0,0])*(t[1,2] - t[1,0]) - (t[1,1] - t[1,0])*(t[0,2] - t[0,0])

    # the are of the triangle is one half the magnitude
    return 0.5*sqrt(x*x + y*y + z*z)


def det(double a0, double a1, double a2, double b0, double b1, double b2, double c0, double c1, double c2):
    return a0*b1*c2 + a1*b2*c0 + a2*b0*c1 - a2*b1*c0 - a1*b0*c2 - a0*b2*c1


def norm(double a0, double a1, double a2, double b0, double b1, double b2, double c0, double c1, double c2,
        double[:] result):

    cdef double x, y, z
    cdef double mag

    x = det(1.0, a1, a2, 1.0, b1, b2, 1.0, c1, c2)
    y = det(a0, 1.0, a2, b0, 1.0, b2, c0, 1.0, c2)
    z = det(a0, a1, 1.0, b0, b1, 1.0, c0, c1, 1.0)

    mag = sqrt(x*x + y*y + z*z)

    result[0] = x/mag
    result[1] = y/mag
    result[2] = z/mag

def cell_face_info_3d(double[:,::1] particles, int[:] neighbor_graph, int[:] num_neighbors,
        int[:] face_graph, int[:] num_face_verts, double[:,::1] voronoi_verts,
        double[:] volume, double[:,::1] center_of_mass,
        double[:] face_areas, double[:,::1] normal, int[:,::1] face_pairs, double[:,::1] face_com,
        int num_real_particles):

    """
    Purpose:
        compute the area and center of mass of each face in 3d

    Discussion:
        The center of mass of the face is the are-weighted sum of the center of mass of
        disjoint triangles that make up the face. Likewise the area of the face is the
        sum of areas of the disjoint triangles.

    Parameters:
        Input, 3xn array of particles positions. Each column is a particle and the rows
        are the x, y, z dimension.

        Input, 1d array of neighbor indices. All neighbors of particle 0 are placed at
        the beginning of the array followed by all the neighbors of particle 1 and so
        on.

        Input, 1d array of number of neighbors. The first value is the number of neighbors
        for particle 0 followed by the number of neighbors for particle 1 and so on. These
        values used to stride the neighbors array.

        Input, 1d array of indices that make up the voronoi faces. For particle 0 all the
        faces are randomly grouped. Then for each face the indices are assigned to this
        array and then repeated for the rest of the particles.

        Input, 1d array of number of vertices that make up each face.

        Input, mx3 array of voronoi positions. Each row corresponds to a particle and
        the columns are the x, y, z and dimension.

        Output, 1d array of volumes for each particle.

        Output, 3xn array of center of mass of voronoi cell. Each column is a particle and
        the rows are x, y, and z dimension.

        Input, the number of real particles.

    Author:
        Ricardo Fernandez

    Reference:
        www.people.sc.fsu.edu/~jburkardt/c_src/geometry/geometry.c
    """

    cdef int id_p      # particle id 
    cdef int id_n      # neighbor id 
    cdef int id_v      # voronoi id 

    cdef int ind_n     # neighbor index
    cdef int ind_f     # face vertex index
    cdef int fi        # face index

    cdef int j, k, p
    cdef double xp, yp, zp, xn, yn, zn
    cdef double vol, area, tri_area, h

    cdef double[:]   com = np.zeros(3,dtype=np.float64)
    cdef double[:,:] tri = np.zeros((3,3),dtype=np.float64)

    cdef int p1, p2, p3
    cdef double[:] n = np.zeros(3,dtype=np.float64)

    fi = 0
    ind_n = 0
    ind_f = 0

    # loop over real particles
    for id_p in range(num_real_particles):

        # get particle position 
        xp = particles[0,id_p]
        yp = particles[1,id_p]
        zp = particles[2,id_p]

        # loop over neighbors
        for j in range(num_neighbors[id_p]):

            # index of neighbor
            id_n = neighbor_graph[ind_n]

            # neighbor position
            xn = particles[0,id_n]
            yn = particles[1,id_n]
            zn = particles[2,id_n]

            # distance from particle to face
            h = 0.5*sqrt((xn-xp)*(xn-xp) + (yn-yp)*(yn-yp) + (zn-zp)*(zn-zp))

            # calculate area of the face between particle and neighbor 

            # area and center of mass of face
            area = 0
            com[0] = com[1] = com[2] = 0

            # last vertex of face
            j = face_graph[num_face_verts[ind_n] - 1 + ind_f]

            # there are n vertices that make up the face but we need
            # only to loop through n-2 vertices

            # decompose polygon into n-2 triangles
            for k in range(num_face_verts[ind_n]-2):

                # index of face vertice
                ind_v = face_graph[ind_f]

                # form a triangle from three vertices
                tri[0,0] = voronoi_verts[ind_v,0]
                tri[1,0] = voronoi_verts[ind_v,1]
                tri[2,0] = voronoi_verts[ind_v,2]

                p = face_graph[ind_f+1]
                tri[0,1] = voronoi_verts[p,0]
                tri[1,1] = voronoi_verts[p,1]
                tri[2,1] = voronoi_verts[p,2]

                tri[0,2] = voronoi_verts[j,0]
                tri[1,2] = voronoi_verts[j,1]
                tri[2,2] = voronoi_verts[j,2]

                # calcualte area of the triangle
                tri_area = triangle_area(tri)

                # face area is the sum of all triangle areas
                area += tri_area

                # the center of mass of the face is the weighted sum of center mass
                # of triangles, the center of mass of a triange is the mean of its vertices
                com[0] += tri_area*(tri[0,0] + tri[0,1] + tri[0,2])/3.0
                com[1] += tri_area*(tri[1,0] + tri[1,1] + tri[1,2])/3.0
                com[2] += tri_area*(tri[2,0] + tri[2,1] + tri[2,2])/3.0

                # skip to next vertex in face
                ind_f += 1

            # skip the remaining two vertices of the face
            ind_f += 2

            # complete the weighted sum for the face
            com[0] /= area
            com[1] /= area
            com[2] /= area

            # the volume of the cell is a collection pyramids 
            # the volume of a pyramid
            vol = area*h/3

            # center of mass of the cell is the sum weighted center of mass of pyramids
            center_of_mass[0,id_p] += vol*(3*com[0] + xp)/4
            center_of_mass[1,id_p] += vol*(3*com[1] + yp)/4
            center_of_mass[2,id_p] += vol*(3*com[2] + zp)/4

            # volume is sum of of pyrmaids
            volume[id_p] += vol






            # expermintal 
            # store face information
            if id_p < id_n:

                # grab the last three points to construct two vectors to make the normal of the face
                p3 = face_graph[ind_f-1]; p2 = face_graph[ind_f-2]; p1 = face_graph[ind_f-3]

                # calculate the normal of the face
                norm(voronoi_verts[p1,0], voronoi_verts[p1,1], voronoi_verts[p1,2],
                        voronoi_verts[p2,0], voronoi_verts[p2,1], voronoi_verts[p2,2],
                        voronoi_verts[p3,0], voronoi_verts[p3,1], voronoi_verts[p3,2], n)

                # store the area of the face
                face_areas[fi] = area

                # store the normal vector of the face
                normal[0,fi] = n[0]
                normal[1,fi] = n[1]
                normal[2,fi] = n[2]

                # store the center mass of the face
                face_com[0,fi] = com[0]
                face_com[1,fi] = com[1]
                face_com[2,fi] = com[2]

                # store the particles that make up the face
                face_pairs[0,fi] = id_p
                face_pairs[1,fi] = id_n

                # go to next face 
                fi += 1






            # go to next neighbor
            ind_n += 1

        # complete the weighted sum for the cell
        center_of_mass[0,id_p] /= volume[id_p]
        center_of_mass[1,id_p] /= volume[id_p]
        center_of_mass[2,id_p] /= volume[id_p]
