
cdef extern from "tess.h":
    cdef cppclass Tess2d:
        Tess2d() except +
        void reset_tess()
        void build_initial_tess(double *x, double *y, double *radius_sq, int num_particles)
        void update_initial_tess(double *x, double *y, int up_num_particles)
        int  count_number_of_faces()
        int extract_geometry(double* x, double* y, double* center_of_mass_x, double* center_of_mass_y,
                double* volume, double* face_area, double* face_nx, double* face_ny, int* pair_i, int* pair_j)
