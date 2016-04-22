from containers.containers cimport ParticleContainer, CarrayContainer
from boundary.boundary cimport BoundaryParallelBase

cdef extern from "tess.h":
    cdef cppclass Tess2d:
        Tess2d() except +
        void reset_tess()
        int build_initial_tess(double *x, double *y, double *radius_sq, int num_particles)
        int update_initial_tess(double *x, double *y, int up_num_particles)
        int count_number_of_faces()
        int extract_geometry(double* x, double* y, double* center_of_mass_x, double* center_of_mass_y, double* volume,
                double* face_area, double* face_comx, double* face_comy, double* face_nx, double* face_ny,
                int* pair_i, int* pair_j)

    cdef cppclass Tess3d:
        Tess3d()
        void reset_tess()
        int build_initial_tess(double *x, double *y, double *z, double *radius_sq, int num_particles, double huge)
        int update_initial_tess(double *x, double *y, double *z, int up_num_particles)
        int count_number_of_faces()
        int extract_geometry(double* x, double* y, double* z, double* center_of_mass_x, double* center_of_mass_y, double* center_of_mass_z,
                double* volume,
                double* face_area, double* face_comx, double* face_comy, double* face_comz,
                double* face_nx, double* face_ny, double* face_nz,
                int* pair_i, int* pair_j)

cdef class MeshBase:

    cdef public ParticleContainer particles
    cdef public BoundaryParallelBase boundary
    cdef public CarrayContainer faces
    cdef public int dim

    cdef _tessellate(self)
    cdef _build_geometry(self)

cdef class Mesh2d(MeshBase):
    cdef Tess2d tess

cdef class Mesh3d(MeshBase):
    cdef Tess3d tess

