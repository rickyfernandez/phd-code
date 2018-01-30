#ifndef __PARTICLE_H__
#define __PARTICLE_H__

#include <list>

struct FlagParticle{
    double x[3];
    double v[3];
    int index;
    double old_search_radius;
    double search_radius;
};

struct GhostID{
    public:
        int index;
        int proc;
        int export_num;

    GhostID() {
        index = 0;
        proc = -1;
        export_num = -1;
    }

    GhostID(const int _index, const int _proc,
            const int _export_num) {
        index = _index;
        proc = _proc;
        export_num = _export_num;
    }
};

struct BoundaryParticle{
    public:
        double x[3];
        double v[3];
        int proc;
        int index;

    BoundaryParticle(const double _x[3], const double _v[3],
            const int _index, const int _proc, const int dim) {

        for(int i=0; i<dim; i++) {
            x[i] = _x[i];
            v[i] = _v[i];
        }
        proc  = _proc;
        index = _index;
    }
};

FlagParticle* particle_flag_deref(std::list<FlagParticle>::iterator &it);

#endif
