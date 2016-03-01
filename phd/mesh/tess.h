#ifndef __MESH_H__
#define __MESH_H__

class Tess2d {
    private:
        int local_num_particles;
        int tot_num_particles;

        void *ptess;
        void *pvt_list;

    public:
        Tess2d(void);
        void reset_tess(void);
        int build_initial_tess(double *x, double *y, double *radius_sq, int num_particles); 
        int update_initial_tess(double *x, double *y, int up_num_particles);
        int count_number_of_faces(void);
        int extract_geometry(double* x, double* y, double* center_of_mass_x, double* center_of_mass_y, double* volume,
                double* face_area, double* face_comx, double* face_comy, double* face_nx, double* face_ny, int* pair_i, int* pair_j);
};

#endif
