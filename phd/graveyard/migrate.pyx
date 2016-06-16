import numpy as np
cimport numpy as np

from ..containers.containers cimport CarrayContainer, ParticleContainer
from ..utils.carray cimport BaseArray, IntArray, LongArray
from cpython cimport PyDict_Contains, PyDict_GetItem

def transfer_migrate_particles(ParticleContainer particles, ParticleContainer import_particles):

    cdef IntArray tags  = particles.get_carray("tag")
    cdef LongArray ids1 = particles.get_carray("ids")
    cdef LongArray proc = particles.get_carray("process")

    cdef LongArray ids2 = import_particles.get_carray("ids")

    cdef int num_real_particles, num_particles
    cdef int i, j, id_p, particle_id

    num_real_particles = particles.num_real_particles
    num_particles = particles.get_number_of_particles()

    for j in range(import_particles.get_number_of_particles()):
        id_p = -1
        particle_id = ids2.data[j]
        for i in range(num_real_particles, num_particles):
            if ids1.data[i] == particle_id:
                # should never happen
                if proc[i] != -1:
                    id_p = i
                    copy_particle(particles, i, import_particles, j)
                    break
        if id_p == -1:
            raise RuntimeError("import particle not found")

cdef copy_particle(ParticleContainer pc1, int i, ParticleContainer pc2, int j):

    cdef str prop_name
    cdef BaseArray dest, source
    cdef np.ndarray nparr_dest, nparr_source

    for prop_name in pc2.properties.keys():
        if PyDict_Contains(pc1.properties, prop_name):
            dest   = <BaseArray> PyDict_GetItem(pc1.properties, prop_name)
            source = <BaseArray> PyDict_GetItem(pc2.properties, prop_name)
            nparr_source = source.get_npy_array()
            nparr_dest = dest.get_npy_array()
            nparr_dest[i] = nparr_source[j]
