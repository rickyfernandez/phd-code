from utils.particle_tags import ParticleTAGS
import numpy as np

from containers.containers cimport CarrayContainer, ParticleContainer
from utils.carray cimport DoubleArray, IntArray, LongLongArray
from libc.math cimport sqrt, atan2, sin, cos
cimport numpy as np
cimport cython


cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost
cdef int Boundary = ParticleTAGS.Boundary

def flag_boundary_particles(ParticleContainer particles, np.int32_t[:] neighbor_graph,
        np.int32_t[:] num_neighbors, np.int32_t[:] cumsum_neighbors):

    cdef IntArray tags = particles.get_carray("tag")
    cdef IntArray type = particles.get_carray("type")

    cdef int i, j
    cdef np.int32_t size, start, id_n

    # determine the number of faces 
    for i in range(particles.get_number_of_particles()):
        if tags.data[i] == Ghost:

            size  = num_neighbors[i]
            start = cumsum_neighbors[i] - size

            # loop over its neighbors
            for j in range(size):

                # index of neighbor
                id_n = neighbor_graph[start + j]

                # check if ghost has a real neighbor
                if tags.data[id_n] == Real:
                    # ghost is a boundary particle
                    type.data[i] = Boundary; break



def number_of_faces(ParticleContainer particles, np.int32_t[:] neighbor_graph, np.int32_t[:] num_neighbors):

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

            for j in range(num_neighbors[id_p]):

                # index of neighbor
                id_n = neighbor_graph[ind]
                if id_n > id_p:
                    num_faces += 1

                # go to next neighbor
                ind += 1

    return num_faces

def cell_face_info_2d(ParticleContainer particles, CarrayContainer faces, np.int32_t[:] neighbor_graph, np.int32_t[:] num_neighbors,
        np.int32_t[:] face_graph, double[:,::1] circum_centers):

    cdef IntArray tags = particles.get_carray("tag")
    cdef IntArray type = particles.get_carray("type")

    cdef DoubleArray x   = particles.get_carray("position-x")
    cdef DoubleArray y   = particles.get_carray("position-y")
    cdef DoubleArray cx  = particles.get_carray("com-x")
    cdef DoubleArray cy  = particles.get_carray("com-y")
    cdef DoubleArray vol = particles.get_carray("volume")

    # face information
    cdef DoubleArray area     = faces.get_carray("area")
    cdef DoubleArray fcx      = faces.get_carray("com-x")
    cdef DoubleArray fcy      = faces.get_carray("com-y")
    cdef DoubleArray nx       = faces.get_carray("normal-x")
    cdef DoubleArray ny       = faces.get_carray("normal-y")
    cdef LongLongArray pair_i = faces.get_carray("pair-i")
    cdef LongLongArray pair_j = faces.get_carray("pair-j")

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
