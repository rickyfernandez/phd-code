#ifndef __PARTICLE_H__
#define __PARTICLE_H__

//struct Particle{
//    public:
//        double x[3];
//        double v[3];
//
//    Particle(const double _x[3], const double _v[3], int dim) {
//        for(int i=0; i<dim; i++) {
//            x[i] = _x[i];
//            v[i] = _v[i];
//        }
//    }
//};

struct QueryParticle{
    public:
        double x[3];
        double v[3];
        double old_radius;
        double new_radius;
        int index;

    QueryParticle(const double _x[3], const double _v[3],
            const double old_radius, const double _new_radius,
            const int _index, const int dim) {
        for(int i=0; i<dim; i++) {
            x[i] = _x[i];
            v[i] = _v[i];
        }
        old_radius = _old_radius;
        new_radius = _new_radius;
        index      = _index;
    }
};

struct BoundaryParticle{
    public:
        double x[3];
        double v[3];

        int index;
        int proc;

    QueryParticle(const double _x[3], const double _v[3],
            const int _proc, const int _index, int dim) {
        for(int i=0; i<dim; i++) {
            x[i] = _x[i];
            v[i] = _v[i];
        }
        proc  = _proc;
        index = _index;
    }
};

#endif
