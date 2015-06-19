from libc.math cimport sqrt, pow, fabs
import numpy as np
cimport numpy as np
cimport cython


def update(double[:,::1] field_data, double[:,::1] fluxes, int[:,::1] face_pairs, double[:] area,
        double dt, int num_faces, int num_real_particles, int num_fields):

    cdef int i, j, k
    cdef double a

    for k in range(num_faces):

        i = face_pairs[0,k]
        j = face_pairs[1,k]
        a = area[k]

        for n in range(num_fields):
            field_data[n,i] -= dt*a*fluxes[n,k]

        if j < num_real_particles:
            for n in range(num_fields):
                field_data[n,j] += dt*a*fluxes[n,k]
