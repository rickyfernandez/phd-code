#ifndef __MESH_H__
#define __MESH_H__

#include <cmath>
#include <vector>
#include <list>
#include "particle.h"

struct vector3 {
    public:
        double x, y, z;
        
        vector3(const double& _x, const double& _y, const double& _z) {
            x = _x; y = _y; z = _z;
        }
        ~vector3(){}

        double norm2() const {
            return (*this)*(*this);
        }
        double abs() const {
            return std::sqrt(norm2());
        }
        const vector3 operator* (const double &s) const {
            return vector3(x*s, y*s, z*s);
        }
        const double operator* (const vector3 &v) const {
            return (x*v.x + y*v.y + z*v.z);
        }
        const vector3 operator+ (const vector3 &v) const {
            return vector3(x + v.x, y + v.y, z + v.z);
        }
        const vector3 operator- (const vector3 &v) const {
            return vector3(x - v.x, y - v.y, z - v.z);
        }
        const vector3 operator = (const vector3 &v) {
            x = v.x; y = v.y; z = v.z;
            return *this;
        }
        const vector3 &operator += (const vector3 &v) {
            *this = *this + v;
            return *this;
        }
        const vector3 &operator *= (const double &s) {
            *this = *this * s;
            return *this;
        }
        const vector3 cross(const vector3 &v) const {
            return vector3(y*v.z - z*v.y, z*v.x  - x*v.z, x*v.y - y*v.x);
        }
};

class Tess2d {
    private:
        int local_num_particles;
        int total_num_particles;

        void *ptess;
        void *pvt_list;

    public:
        Tess2d(void);
        void reset_tess(void);
        int build_initial_tess(double *x[3], double *radius, int num_real_particles); 
        int update_initial_tess(double *x[3], int begin_particles, int end_particles);
        int count_number_of_faces(void);
        int extract_geometry(double* x[3], double* dcom[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3], int* pair_i, int* pair_j,
                std::vector< std::vector<int> > &neighbors);
        int update_radius(double *x[3], double *radius, std::list<FlagParticle> &flagged_particles);
        int reindex_ghost(std::vector<GhostID> &import_ghost_buffer);
};

class Tess3d {
    private:
        int local_num_particles;
        int total_num_particles;

        void *ptess;
        void *pvt_list;

    public:
        Tess3d(void);
        void reset_tess(void);
        int build_initial_tess(double *x[3], double *radius, int num_real_particles); 
        int update_initial_tess(double *x[3], int begin_particles, int end_particles);
        int count_number_of_faces(void);
        int extract_geometry(double* x[3], double* dcom[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3], int* pair_i, int* pair_j,
                std::vector< std::vector<int> > &neighbors);
        int update_radius(double *x[3], double *radius, std::list<FlagParticle> &flagged_particles);
        int reindex_ghost(std::vector<GhostID> &import_ghost_buffer);
};

#endif
