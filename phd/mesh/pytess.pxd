from libcpp.vector cimport vector
from libcpp.list cimport list as cpplist

from ..domain.domain_manager cimport FlagParticle

ctypedef vector[int] nn
ctypedef vector[nn] nn_vec

cdef extern from "tess.h":
    cdef cppclass Tess2d:
        Tess2d() except +
        void reset_tess()
        int build_initial_tess(double *x[3], double *radius, int num_real_particles)
        int update_initial_tess(double *x[3], int begin_particles, int end_particles)
        int count_number_of_faces()
        int extract_geometry(double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j, nn_vec &neighbors)
        int update_radius(double *x[3], double *radius, cpplist[FlagParticle] flagged_particles)

    cdef cppclass Tess3d:
        Tess3d() except +
        void reset_tess()
        int build_initial_tess(double *x[3], double *radius, int num_real_particles)
        int update_initial_tess(double *x[3], int begin_particles, int end_particles)
        int count_number_of_faces()
        int extract_geometry(double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j, nn_vec &neighbors)
        int update_radius(double *x[3], double *radius, cpplist[FlagParticle] flagged_particles)

cdef class PyTess:

    cdef void reset_tess(self)
    cdef int build_initial_tess(self, double *x[3], double *radius, int num_real_particles)
    cdef int update_initial_tess(self, double *x[3], int begin_particles, int end_particles)
    cdef int count_number_of_faces(self)
    cdef int extract_geometry(self, double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j, nn_vec &neighbors)
    cdef int update_radius(self, double *x[3], double *radius, cpplist[FlagParticle] flagged_particles)

cdef class PyTess2d(PyTess):
    cdef Tess2d *thisptr

cdef class PyTess3d(PyTess):
    cdef Tess3d *thisptr
