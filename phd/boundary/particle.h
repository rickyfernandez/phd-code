#ifndef __PARTICLE_H__
#define __PARTICLE_H__

struct Particle{
    public:
        double x[3];
        double v[3];

    Particle(const double _x[3], const double _v[3], int dim) {
        for(int i=0; i<dim; i++) {
            x[i] = _x[i];
            v[i] = _v[i];
        }
    }
};

#endif
