from utils.particle_tags import ParticleTAGS
import numpy as np

from containers.containers cimport CarrayContainer, ParticleContainer
from utils.carray cimport DoubleArray, IntArray
from libc.math cimport sqrt, atan2, sin, cos
cimport numpy as np
cimport cython


cdef int Real = ParticleTAGS.Real
cdef int Boundary = ParticleTAGS.Boundary

def number_of_faces(ParticleContainer particles, int[:] neighbor_graph, int[:] neighbor_graph_size):

    cdef IntArray tags = particles.get_carray("tag")
    cdef IntArray type = particles.get_carray("type")

    cdef int num_faces
    cdef int id_p, id_n
    cdef int ind, j

    ind = 0
    num_faces = 0

    # determine the number of faces 
    for id_p in range(particles.get_number_of_particles()):

        if tags.data[id_p] == Real or type.data[id_p] == Boundary:

            for j in range(neighbor_graph_size[id_p]):

                # index of neighbor
                id_n = neighbor_graph[ind]
                if id_n > id_p:
                    num_faces += 1

                # go to next neighbor
                ind += 1

    return num_faces

def cell_face_info_2d(ParticleContainer particles, CarrayContainer faces, int[:] neighbor_graph, int[:] num_neighbors,
        int[:] face_graph, double[:,::1] circum_centers):

    cdef IntArray tags = particles.get_carray("tag")
    cdef IntArray type = particles.get_carray("type")

    cdef DoubleArray x   = particles.get_carray("position-x")
    cdef DoubleArray y   = particles.get_carray("position-y")
    cdef DoubleArray cx  = particles.get_carray("com-x")
    cdef DoubleArray cy  = particles.get_carray("com-y")
    cdef DoubleArray vol = particles.get_carray("volume")

    # face information
    cdef DoubleArray area   = faces.get_carray("area")
    cdef DoubleArray fcx    = faces.get_carray("com-x")
    cdef DoubleArray fcy    = faces.get_carray("com-y")
    cdef DoubleArray nx     = faces.get_carray("normal-x")
    cdef DoubleArray ny     = faces.get_carray("normal-y")
    cdef DoubleArray pair_i = faces.get_carray("pair-i")
    cdef DoubleArray pair_j = faces.get_carray("pair-j")

    cdef int id_p          # particle index 
    cdef int id_n          # neighbor index 
    cdef int ind           # loop index
    cdef int ind_face      # face vertex index

    cdef int j, k

    cdef double _xp, _yp, _xn, _yn, _x1, _y1, _x2, _y2, _xr, _yr, _x, _y
    cdef double face_area, h

    cdef double _fx, _fy, _tx, _ty

    k = 0
    ind = 0
    ind_face = 0

    # loop over real particles
    for id_p in range(particles.get_number_of_particles()):

        if tags.data[id_p] == Real or type.data[id_p] == Boundary:

            # get poistion of particle
            _xp = x.data[id_p]
            _yp = y.data[id_p]

            # loop over its neighbors
            for j in range(num_neighbors[id_p]):

                # index of neighbor
                id_n = neighbor_graph[ind]

                # neighbor position
                _xn = x.data[id_n]
                _yn = y.data[id_n]

                # difference vector between particles 
                _xr = _xn - _xp
                _yr = _yn - _yp

                # distance between particles
                h = sqrt(_xr*_xr + _yr*_yr)

                # calculate area of face between particle and neighbor 
                # in 2d each face is made up of two vertices
                _x1 = circum_centers[face_graph[ind_face],0]
                _y1 = circum_centers[face_graph[ind_face],1]
                ind_face += 1 # go to next face vertex

                _x2 = circum_centers[face_graph[ind_face],0]
                _y2 = circum_centers[face_graph[ind_face],1]
                ind_face += 1 # go to next face

                # edge vector
                _x = _x2 - _x1
                _y = _y2 - _y1

                # face area in 2d is length between voronoi vertices  
                face_area = sqrt(_x*_x + _y*_y)

                # the volume of the cell is the sum of triangle areas - eq. 27
                vol.data[id_p] += 0.25*face_area*h

                # center of mass of face
                _fx = 0.5*(_x1 + _x2)
                _fy = 0.5*(_y1 + _y2)

                # center of mass of a triangle - eq. 31
                _tx = 2.0*_fx/3.0 + _xp/3.0
                _ty = 2.0*_fy/3.0 + _yp/3.0

                # center of mass of the cell is the sum weighted center of mass of
                # the triangles - eq. 29
                cx.data[id_p] += 0.25*face_area*h*_tx
                cy.data[id_p] += 0.25*face_area*h*_ty

                # store face information
                if id_p < id_n:

                    # store the area of the face
                    area.data[k] = face_area

                    # store the orientation of the norm of the face
                    nx.data[k] = _xr/h
                    ny.data[k] = _yr/h

                    # store the center mass of the face
                    fcx.data[k] = _fx
                    fcy.data[k] = _fy

                    # store the particles that make up the face
                    pair_i.data[k] = id_p
                    pair_j.data[k] = id_n

                    # go to next face 
                    k += 1

                # go to next neighbor
                ind += 1

            # compelte the weighted sum for the cell - eq. 29
            cx.data[id_p] /= vol.data[id_p]
            cy.data[id_p] /= vol.data[id_p]


#@cython.boundscheck(False)
#@cython.wraparound(False)
#def assign_face_velocities_2d(double[:,::1] particles, int[:] neighbor_graph, int[:] num_neighbors,
#        double[:,::1] face_com, double[:,::1] face_velocities, double[:,::1] w, int num_particles):
#
#    cdef int id_p, id_n, ind, j, k
#    cdef double xp, yp, xn, yn
#    cdef double factor
#
#    k = 0
#    ind = 0
#
#    for id_p in range(num_particles):
#
#        # position of particle
#        xp = particles[0,id_p]
#        yp = particles[1,id_p]
#
#        for j in range(num_neighbors[id_p]):
#
#            # index of neighbor
#            id_n = neighbor_graph[ind]
#
#            if id_p < id_n:
#
#                # position of neighbor
#                xn = particles[0,id_n]
#                yn = particles[1,id_n]
#
#                # the face velocity is approx the mean of velocities  of the particles
#                face_velocities[0,k] = 0.5*(w[0, id_p] + w[0, id_n])
#                face_velocities[1,k] = 0.5*(w[1, id_p] + w[1, id_n])
#
#                # coordinates of face center of mass
#                fx = face_com[0,k]
#                fy = face_com[1,k]
#
#                # correct face velocity due to residual motion - eq. 33
#                factor  = (w[0,id_p]-w[0,id_n])*(fx-0.5*(xp+xn)) + (w[1,id_p]-w[1,id_n])*(fy-0.5*(yp+yn))
#                factor /= (xn-xp)*(xn-xp) + (yn-yp)*(yn-yp)
#
#                # store the face velocity
#                face_velocities[0,k] += factor*(xn-xp)
#                face_velocities[1,k] += factor*(yn-yp)
#
#                # update counter
#                k   += 1
#                ind += 1
#
#            else:
#
#                # face accounted for, go to next neighbor and face
#                ind += 1




#def assign_face_velocities_3d(double[:,::1] particles, int[:] neighbor_graph, int[:] num_neighbors,
#        double[:,::1] face_com, double[:,::1] face_velocities, double[:,::1] w, int num_particles):
#
#    cdef int id_p, id_n, ind, j, k
#    cdef double xp, yp, zp, xn, yn, zn
#    cdef double fx, fy, fz
#    cdef double factor
#
#    k = 0
#    ind = 0
#
#    for id_p in range(num_particles):
#
#        # position of particle
#        xp = particles[0,id_p]
#        yp = particles[1,id_p]
#        zp = particles[2,id_p]
#
#        for j in range(num_neighbors[id_p]):
#
#            # index of neighbor
#            id_n = neighbor_graph[ind]
#
#            if id_p < id_n:
#
#                # position of neighbor
#                xn = particles[0,id_n]
#                yn = particles[1,id_n]
#                zn = particles[2,id_n]
#
#                # the face velocity is approx the mean of velocities of the particles
#                face_velocities[0,k] = 0.5*(w[0, id_p] + w[0, id_n])
#                face_velocities[1,k] = 0.5*(w[1, id_p] + w[1, id_n])
#                face_velocities[2,k] = 0.5*(w[2, id_p] + w[2, id_n])
#
#                # coordinates of face center of mass
#                fx = face_com[0,k]
#                fy = face_com[1,k]
#                fz = face_com[2,k]
#
#                # correct face velocity due to residual motion - eq. 33
#                factor  = (w[0,id_p]-w[0,id_n])*(fx-0.5*(xp+xn)) + (w[1,id_p]-w[1,id_n])*(fy-0.5*(yp+yn)) + (w[2,id_p]-w[2,id_n])*(fz-0.5*(zp+zn))
#                factor /= (xn-xp)*(xn-xp) + (yn-yp)*(yn-yp) + (zn-zp)*(zn-zp)
#
#                face_velocities[0,k] += factor*(xn-xp)
#                face_velocities[1,k] += factor*(yn-yp)
#                face_velocities[2,k] += factor*(zn-zp)
#
#                # update counter
#                k   += 1
#                ind += 1
#
#            else:
#
#                # face accounted for, go to next neighbor and face
#                ind += 1


#def triangle_area(double[:,::1] t):
#    """
#    Purpose:
#        compute the area of a triangle 3d
#
#    Discussion:
#        The area of a parallelogram is the cross product of two sides of the parallelogram.
#        The triangle formed from the parallelogram is half the area
#
#    Parameters:
#        Input, 3x3 array of vertices of the triangle. Each column is a point and the rows
#        are the x, y, and z dimension.
#
#        Output, the area of the triangle.
#
#    Author:
#        Ricardo Fernandez
#
#    Reference:
#        www.people.sc.fsu.edu/~jburkardt/c_src/geometry/geometry.c
#    """
#
#    cdef double x, y, z
#
#    # perform cross product from the vertices that make up the triangle
#    # (v1 - v0) x (v2 - v0)
#    x = (t[1,1] - t[1,0])*(t[2,2] - t[2,0]) - (t[2,1] - t[2,0])*(t[1,2] - t[1,0])
#    y = (t[2,1] - t[2,0])*(t[0,2] - t[0,0]) - (t[0,1] - t[0,0])*(t[2,2] - t[2,0])
#    z = (t[0,1] - t[0,0])*(t[1,2] - t[1,0]) - (t[1,1] - t[1,0])*(t[0,2] - t[0,0])
#
#    # the are of the triangle is one half the magnitude
#    return 0.5*sqrt(x*x + y*y + z*z)


#def cell_face_info_3d(double[:,::1] particles, int[:] neighbor_graph, int[:] num_neighbors,
#        int[:] face_graph, int[:] num_face_verts, double[:,::1] voronoi_verts,
#        double[:] volume, double[:,::1] center_of_mass,
#        double[:] face_areas, double[:,::1] normal, int[:,::1] face_pairs, double[:,::1] face_com,
#        int num_real_particles):
#
#    """
#    Purpose:
#        compute the area and center of mass of each face in 3d
#
#    Discussion:
#        The center of mass of the face is the are-weighted sum of the center of mass of
#        disjoint triangles that make up the face. Likewise the area of the face is the
#        sum of areas of the disjoint triangles.
#
#    Parameters:
#        Input, 3xn array of particles positions. Each column is a particle and the rows
#        are the x, y, z dimension.
#
#        Input, 1d array of neighbor indices. All neighbors of particle 0 are placed at
#        the beginning of the array followed by all the neighbors of particle 1 and so
#        on.
#
#        Input, 1d array of number of neighbors. The first value is the number of neighbors
#        for particle 0 followed by the number of neighbors for particle 1 and so on. These
#        values used to stride the neighbors array.
#
#        Input, 1d array of indices that make up the voronoi faces. For particle 0 all the
#        faces are randomly grouped. Then for each face the indices are assigned to this
#        array and then repeated for the rest of the particles.
#
#        Input, 1d array of number of vertices that make up each face.
#
#        Input, mx3 array of voronoi positions. Each row corresponds to a particle and
#        the columns are the x, y, z and dimension.
#
#        Output, 1d array of volumes for each particle.
#
#        Output, 3xn array of center of mass of voronoi cell. Each column is a particle and
#        the rows are x, y, and z dimension.
#
#        Input, the number of real particles.
#
#    Author:
#        Ricardo Fernandez
#
#    Reference:
#        www.people.sc.fsu.edu/~jburkardt/c_src/geometry/geometry.c
#    """
#
#    cdef int id_p      # particle id 
#    cdef int id_n      # neighbor id 
#    cdef int id_v      # voronoi id 
#
#    cdef int ind_n     # neighbor index
#    cdef int ind_f     # face vertex index
#    cdef int fi        # face index
#
#    cdef int i, j, k, p
#    cdef double xp, yp, zp, xn, yn, zn
#    cdef double vol, area, tri_area, h
#
#    cdef double[:]   com = np.zeros(3,dtype=np.float64)
#    cdef double[:,:] tri = np.zeros((3,3),dtype=np.float64)
#
#    cdef double[:] n = np.zeros(3,dtype=np.float64)
#
#    fi = 0
#    ind_n = 0
#    ind_f = 0
#
#    # loop over real particles
#    for id_p in range(num_real_particles):
#
#        # get particle position 
#        xp = particles[0,id_p]
#        yp = particles[1,id_p]
#        zp = particles[2,id_p]
#
#        # loop over neighbors
#        for i in range(num_neighbors[id_p]):
#
#            # index of neighbor
#            id_n = neighbor_graph[ind_n]
#
#            # neighbor position
#            xn = particles[0,id_n]
#            yn = particles[1,id_n]
#            zn = particles[2,id_n]
#
#            # distance from particle to face
#            h = 0.5*sqrt((xn-xp)*(xn-xp) + (yn-yp)*(yn-yp) + (zn-zp)*(zn-zp))
#
#            # calculate area of the face between particle and neighbor 
#
#            # area and center of mass of face
#            area = 0
#            com[0] = com[1] = com[2] = 0
#
#            # last vertex of face
#            j = face_graph[num_face_verts[ind_n] - 1 + ind_f]
#
#            tri[0,2] = voronoi_verts[j,0]
#            tri[1,2] = voronoi_verts[j,1]
#            tri[2,2] = voronoi_verts[j,2]
#
#            # there are n vertices that make up the face but we need
#            # only to loop through n-2 vertices
#
#            # decompose polygon into n-2 triangles
#            for k in range(num_face_verts[ind_n]-2):
#
#                # index of face vertice
#                ind_v = face_graph[ind_f]
#
#                # form a triangle from three vertices
#                tri[0,0] = voronoi_verts[ind_v,0]
#                tri[1,0] = voronoi_verts[ind_v,1]
#                tri[2,0] = voronoi_verts[ind_v,2]
#
#                p = face_graph[ind_f+1]
#                tri[0,1] = voronoi_verts[p,0]
#                tri[1,1] = voronoi_verts[p,1]
#                tri[2,1] = voronoi_verts[p,2]
#
#                # calcualte area of the triangle
#                tri_area = triangle_area(tri)
#
#                # face area is the sum of all triangle areas
#                area += tri_area
#
#                # the center of mass of the face is the weighted sum of center mass
#                # of triangles, the center of mass of a triange is the mean of its vertices
#                com[0] += tri_area*(tri[0,0] + tri[0,1] + tri[0,2])/3.0
#                com[1] += tri_area*(tri[1,0] + tri[1,1] + tri[1,2])/3.0
#                com[2] += tri_area*(tri[2,0] + tri[2,1] + tri[2,2])/3.0
#
#                # skip to next vertex in face
#                ind_f += 1
#
#            # skip the remaining two vertices of the face
#            ind_f += 2
#
#            # complete the weighted sum for the face
#            com[0] /= area
#            com[1] /= area
#            com[2] /= area
#
#            # the volume of the cell is a collection pyramids 
#            # the volume of a pyramid
#            vol = area*h/3
#
#            # center of mass of the cell is the sum weighted center of mass of pyramids
#            center_of_mass[0,id_p] += vol*(3*com[0] + xp)/4
#            center_of_mass[1,id_p] += vol*(3*com[1] + yp)/4
#            center_of_mass[2,id_p] += vol*(3*com[2] + zp)/4
#
#            # volume is sum of of pyrmaids
#            volume[id_p] += vol
#
#
#
#
#
#
#            # expermintal 
#            # store face information
#            if id_p < id_n:
#
#                # difference vector between particles
#                xr = xn - xp
#                yr = yn - yp
#                zr = zn - zp
#
#                # calculate face normal
#                mag = sqrt(xr*xr + yr*yr + zr*zr)
#                n[0] = xr/mag
#                n[1] = yr/mag
#                n[2] = zr/mag
#
#                if area < 1.0E-15:
#                    print
#                    print "error"
#                    print "probelm with cell:", id_p
#                    print "neighbor:", id_n
#                    print "coordinates:", xp, yp, zp
#                    print "face area:", area
#                    print "number of vertices:", num_face_verts[ind_n]
#                    print
#                    #print "norm:", sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
#                    #print "x norm:", n[0]
#                    #print "y norm:", n[1]
#                    #print "z norm:", n[2], "\n"
#
#                # store the area of the face
#                face_areas[fi] = area
#
#                # store the normal vector of the face
#                normal[0,fi] = n[0]
#                normal[1,fi] = n[1]
#                normal[2,fi] = n[2]
#
#                # store the center mass of the face
#                face_com[0,fi] = com[0]
#                face_com[1,fi] = com[1]
#                face_com[2,fi] = com[2]
#
#                # store the particles that make up the face
#                face_pairs[0,fi] = id_p
#                face_pairs[1,fi] = id_n
#
#                # go to next face 
#                fi += 1
#
#
#
#
#
#
#            # go to next neighbor
#            ind_n += 1
#
#        # complete the weighted sum for the cell
#        center_of_mass[0,id_p] /= volume[id_p]
#        center_of_mass[1,id_p] /= volume[id_p]
#        center_of_mass[2,id_p] /= volume[id_p]
