#include <particle.h>

FlagParticle* particle_flag_deref(std::list<FlagParticle>::iterator &it) {
    return &(*it);
}
