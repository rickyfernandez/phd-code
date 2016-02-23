import numpy as np
cimport numpy as np

from domain.domain cimport DomainLimits
from utils.particle_tags import ParticleTAGS
from utils.carray cimport DoubleArray, LongArray
from containers.containers cimport ParticleContainer, CarrayContainer

cdef int Ghost = ParticleTAGS.Ghost

cdef class BoundaryBase2d:
    def __init__(self, DomainLimits domain):
        self.domain = DomainLimits

    cdef int _create_ghost_particles(ParticleContainer pc):
        pass

cdef class Reflect2d(BoundaryBase2d):

    cdef int create_ghost_particles(ParticleContainer pc):

        cdef CarrayContainer copy
        cdef np.ndarray npy_array

        cdef DoubleArray x = pc.get_carray("position-x")
        cdef DoubleArray y = pc.get_carray("position-y")
        cdef DoubleArray r = pc.get_carray("radius")

        cdef LongArray left = LongArray()
        cdef LongArray right = LongArray()
        cdef LongArray bottom = LongArray()
        cdef LongArray top = LongArray()

        cdef double xmin = domain.xmin
        cdef double xmax = domain.xmax
        cdef double ymin = domain.ymin
        cdef double ymax = domain.ymax

        cdef double x1i, x2i, y1i, y2i

        cdef int i, num_ghost = 0
        for i in range(pc.get_number_of_particles()):

            # bounding box of particle
            x1i = x[i] - r[i]; x2i = x[i] + r[i]
            y1i = y[i] - r[i]; y2i = y[i] + r[i]

            # left boundary condition
            if x1i < xmin:
                left.append(i)

            # right boundary condition
            if xmax < x2i:
                right.append(i)

            # bottom boundary condition
            if y1i < ymin:
                bottom.append(i)

            # top boundary condition
            if ymax < y2i:
                top.append(i)

        # left ghost particles
        npy_array = left.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-x'][:] -= 2*(copy['position-x'] - xmin)
        copy['velocity-x'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        # right ghost particles
        npy_array = right.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-x'][:] -= 2*(copy['position-x'] - xmax)
        copy['velocity-x'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        # bottom ghost particles
        npy_array = bottom.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-y'][:] -= 2*(copy['position-y'] - ymin)
        copy['velocity-y'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        # top ghost particles
        npy_array = top.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-y'][:] -= 2*(copy['position-y'] - ymax)
        copy['velocity-y'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        return num_ghost
