import numpy as np
cimport numpy as np

from domain.domain cimport DomainLimits
from utils.particle_tags import ParticleTAGS
from utils.carray cimport DoubleArray, LongArray
from containers.containers cimport ParticleContainer, CarrayContainer

cdef int Ghost = ParticleTAGS.Ghost

cdef class BoundaryBase:
    def __init__(self, DomainLimits domain):
        self.domain = domain

    cdef int _create_ghost_particles(self, ParticleContainer pc):
        msg = "BoundaryBase::_create_ghost_particles called!"
        raise NotImplementedError(msg)

cdef class Reflect2d(BoundaryBase):

    cdef int _create_ghost_particles(self, ParticleContainer pc):

        cdef CarrayContainer copy
        cdef np.ndarray npy_array

        cdef DoubleArray x = pc.get_carray("position-x")
        cdef DoubleArray y = pc.get_carray("position-y")
        cdef DoubleArray r = pc.get_carray("radius")

        cdef LongArray xlower = LongArray()
        cdef LongArray xupper = LongArray()
        cdef LongArray ylower = LongArray()
        cdef LongArray yupper = LongArray()

        cdef double xmin = self.domain.xmin
        cdef double xmax = self.domain.xmax
        cdef double ymin = self.domain.ymin
        cdef double ymax = self.domain.ymax

        cdef double x_lo, x_hi, y_lo, y_hi

        cdef int i, num_ghost = 0
        for i in range(pc.get_number_of_particles()):

            # bounding box of particle
            x_lo = x[i] - r[i]; x_hi = x[i] + r[i]
            y_lo = y[i] - r[i]; y_hi = y[i] + r[i]

            # left boundary condition
            if x_lo < xmin:
                xlower.append(i)

            # right boundary condition
            if xmax < x_hi:
                xupper.append(i)

            # bottom boundary condition
            if y_lo < ymin:
                ylower.append(i)

            # top boundary condition
            if ymax < y_hi:
                yupper.append(i)

        # left ghost particles
        npy_array = xlower.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-x'][:] -= 2*(copy['position-x'] - xmin)
        copy['velocity-x'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        # right ghost particles
        npy_array = xupper.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-x'][:] -= 2*(copy['position-x'] - xmax)
        copy['velocity-x'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        # bottom ghost particles
        npy_array = ylower.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-y'][:] -= 2*(copy['position-y'] - ymin)
        copy['velocity-y'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        # top ghost particles
        npy_array = yupper.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-y'][:] -= 2*(copy['position-y'] - ymax)
        copy['velocity-y'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        return num_ghost

cdef class Reflect3d(BoundaryBase):

    cdef int _create_ghost_particles(self, ParticleContainer pc):

        cdef CarrayContainer copy
        cdef np.ndarray npy_array

        cdef DoubleArray x = pc.get_carray("position-x")
        cdef DoubleArray y = pc.get_carray("position-y")
        cdef DoubleArray z = pc.get_carray("position-z")
        cdef DoubleArray r = pc.get_carray("radius")

        cdef LongArray xlower = LongArray()
        cdef LongArray xupper = LongArray()
        cdef LongArray ylower = LongArray()
        cdef LongArray yupper = LongArray()
        cdef LongArray zlower = LongArray()
        cdef LongArray zupper = LongArray()

        cdef double xmin = self.domain.xmin
        cdef double xmax = self.domain.xmax
        cdef double ymin = self.domain.ymin
        cdef double ymax = self.domain.ymax
        cdef double zmin = self.domain.zmin
        cdef double zmax = self.domain.zmax

        cdef double x_lo, x_hi, y_lo, y_hi, z_lo, z_hi

        cdef int i, num_ghost = 0
        for i in range(pc.get_number_of_particles()):

            r[i] = min(0.25*self.domain.xtranslate, r[i])

            # bounding box of particle
            x_lo = x[i] - r[i]; x_hi = x[i] + r[i]
            y_lo = y[i] - r[i]; y_hi = y[i] + r[i]
            z_lo = z[i] - r[i]; z_hi = z[i] + r[i]

            # left boundary condition
            if x_lo < xmin:
                xlower.append(i)

            # right boundary condition
            if xmax < x_hi:
                xupper.append(i)

            # bottom boundary condition
            if y_lo < ymin:
                ylower.append(i)

            # top boundary condition
            if ymax < y_hi:
                yupper.append(i)

            # bottom boundary condition
            if z_lo < zmin:
                zlower.append(i)

            # top boundary condition
            if zmax < z_hi:
                zupper.append(i)

        # ghost particles in xmin
        npy_array = xlower.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-x'][:] -= 2*(copy['position-x'] - xmin)
        copy['velocity-x'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        # ghost particles in xmax
        npy_array = xupper.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-x'][:] -= 2*(copy['position-x'] - xmax)
        copy['velocity-x'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        # ghost particles in ymin
        npy_array = ylower.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-y'][:] -= 2*(copy['position-y'] - ymin)
        copy['velocity-y'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        # ghost particles in ymax
        npy_array = yupper.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-y'][:] -= 2*(copy['position-y'] - ymax)
        copy['velocity-y'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        # ghost particles in zmin
        npy_array = zlower.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-z'][:] -= 2*(copy['position-z'] - zmin)
        copy['velocity-z'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()

        # ghost particles in zmax
        npy_array = zupper.get_npy_array()
        copy = pc.extract_items(npy_array)
        copy['position-z'][:] -= 2*(copy['position-z'] - zmax)
        copy['velocity-z'][:] *= -1.0
        copy['tag'][:] = Ghost
        copy['map'][:] = npy_array
        pc.append_container(copy)
        num_ghost += copy.get_number_of_items()
        return num_ghost
