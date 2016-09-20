from libcpp.vector cimport vector

from ..containers.containers cimport ParticleContainer, CarrayContainer
from ..boundary.boundary cimport Boundary

#ctypedef vector[int] nns           # nearest neighbors
#ctypedef vector[neighbors] nns_vec

cdef extern from "tess.h":
    cdef cppclass Tess2d:
        Tess2d() except +
        void reset_tess()
        int build_initial_tess(double *x[3], double *radius_sq, int num_particles, double huge)
        int update_initial_tess(double *x[3], int up_num_particles)
        int count_number_of_faces()
        int extract_geometry(double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j)

    cdef cppclass Tess3d:
        Tess3d() except +
        void reset_tess()
        int build_initial_tess(double *x[3], double *radius_sq, int num_particles, double huge)
        int update_initial_tess(double *x[3], int up_num_particles)
        int count_number_of_faces()
        int extract_geometry(double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j)

cdef class PyTess:

    cdef void reset_tess(self)
    cdef int build_initial_tess(self, double *x[3], double *radius_sq, int num_particles, double huge)
    cdef int update_initial_tess(self, double *x[3], int up_num_particles)
    cdef int count_number_of_faces(self)
    cdef int extract_geometry(self, double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j)

cdef class PyTess2d(PyTess):
    cdef Tess2d *thisptr

cdef class PyTess3d(PyTess):
    cdef Tess3d *thisptr

cdef class Mesh:

    cdef public Boundary boundary
    cdef public CarrayContainer faces
    cdef public int dim
    cdef public list fields

#    cdef public nns_vec neighbors

    cdef PyTess tess

    cdef _tessellate(self, ParticleContainer pc)
    cdef _build_geometry(self, ParticleContainer pc)
    cdef _reset_mesh(self)
