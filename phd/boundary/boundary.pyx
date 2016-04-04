import numpy as np
cimport numpy as np

from domain.domain cimport DomainLimits
from utils.particle_tags import ParticleTAGS
from utils.carray cimport DoubleArray, LongArray, IntArray
from containers.containers cimport ParticleContainer, CarrayContainer

cdef int Ghost = ParticleTAGS.Ghost

cdef class BoundaryBase:
    def __init__(self, DomainLimits domain):
        self.domain = domain

    cdef int _create_ghost_particles(self, ParticleContainer pc):
        msg = "BoundaryBase::_create_ghost_particles called!"
        raise NotImplementedError(msg)

    cdef _update_ghost_particles(self, ParticleContainer pc):
        msg = "BoundaryBase::_update_ghost_particles called!"
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

            r[i] = min(0.25*self.domain.xtranslate, r[i])

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

    cdef _update_ghost_particles(self, ParticleContainer pc):

        cdef IntArray tags = pc.get_carray("tag")
        cdef LongArray maps = pc.get_carray("map")

        cdef DoubleArray x = pc.get_carray("position-x")
        cdef DoubleArray y = pc.get_carray("position-y")
        cdef DoubleArray comx = pc.get_carray("com-x")
        cdef DoubleArray comy = pc.get_carray("com-y")
        cdef DoubleArray vol = pc.get_carray("volume")

        cdef double xmin = self.domain.xmin
        cdef double xmax = self.domain.xmax
        cdef double ymin = self.domain.ymin
        cdef double ymax = self.domain.ymax

        cdef int i, j

        for i in range(pc.get_number_of_particles()):
            if tags.data[i] == Ghost:

                j = maps.data[i]

                comx.data[i] = comx.data[j]
                comy.data[i] = comy.data[j]
                vol.data[i] = vol.data[j]

                if x.data[i] < xmin:
                    comx.data[i] -= 2*(comx.data[j] - xmin)
                if xmax < x.data[i]:
                    comx.data[i] -= 2*(comx.data[j] - xmax)

                if y.data[i] < ymin:
                    comy.data[i] -= 2*(comy.data[j] - ymin)
                if ymax < y.data[i]:
                    comy.data[i] -= 2*(comy.data[j] - ymax)

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

    cdef _update_ghost_particles(self, ParticleContainer pc):

        cdef IntArray tags = pc.get_carray("tag")
        cdef LongArray maps = pc.get_carray("map")

        cdef DoubleArray x = pc.get_carray("position-x")
        cdef DoubleArray y = pc.get_carray("position-y")
        cdef DoubleArray z = pc.get_carray("position-z")
        cdef DoubleArray comx = pc.get_carray("com-x")
        cdef DoubleArray comy = pc.get_carray("com-y")
        cdef DoubleArray comz = pc.get_carray("com-z")
        cdef DoubleArray vol = pc.get_carray("volume")

        cdef double xmin = self.domain.xmin
        cdef double xmax = self.domain.xmax
        cdef double ymin = self.domain.ymin
        cdef double ymax = self.domain.ymax
        cdef double zmin = self.domain.zmin
        cdef double zmax = self.domain.zmax

        cdef int i, j

        for i in range(pc.get_number_of_particles()):
            if tags.data[i] == Ghost:

                j = maps.data[i]

                comx.data[i] = comx.data[j]
                comy.data[i] = comy.data[j]
                vol.data[i] = vol.data[j]

                if x.data[i] < xmin:
                    comx.data[i] -= 2*(comx.data[j] - xmin)
                if xmax < x.data[i]:
                    comx.data[i] -= 2*(comx.data[j] - xmax)

                if y.data[i] < ymin:
                    comy.data[i] -= 2*(comy.data[j] - ymin)
                if ymax < y.data[i]:
                    comy.data[i] -= 2*(comy.data[j] - ymax)

                if z.data[i] < zmin:
                    comz.data[i] -= 2*(comz.data[j] - zmin)
                if zmax < z.data[i]:
                    comz.data[i] -= 2*(comz.data[j] - zmax)

cdef class PeriodicParallel(BoundaryParallelBase):

    cdef int _create_ghost_particles(self, ParticleContainer pc):

        cdef LongLongArray keys = pc.get_carray("key")

        cdef DoubleArray xg = DoubleArray()
        cdef DoubleArray yg = DoubleArray()
        cdef DoubleArray zg = DoubleArray()

        cdef np.float64_t* x[3]
        cdef int i, j, current, num_neighbors

        self.buffer_ids.reset()
        self.buffer_pid.reset()
        self.num_export = 0

        pc.extract_field_vec_ptr(x, "position")

        for i in range(pc.get_number_of_particles()):

            # first we do interior ghost particles

            # find the size of the leaf where particle lives in
            node = self.load_balance.global_tree._find_leaf(keys.data[i])
            r[i] = min(0.25*node.box_length, r[i])

            # bounding box of particle
            for j in range(self.dim):
                center[j] = x[i][j]

            # find overlaping processors
            nbrs_array.reset()
            num_neighbors = self.load_balance.global_tree._get_nearest_neighbors(
                    center, r[i], self.load_balance.leaf_proc, self.rank, nbrs)
            nbrs_array = nbrs.get_npy_array()

            # put in processors in order to avoid duplicates
            nbrs_array.sort()

            # put particles in buffer
            for current range(num_neighbors):

                if current == num_neighbors-1:
                    if nbrs_array[current] != nbrs_array[current-1]:
                        buffer_ids.append(i)
                        buffer_pid.append(nbrs_array[current])
                        num_export += 1
                else:
                    if nbrs_array[current] != nbrs_array[current+1]:
                        buffer_ids.append(i)
                        buffer_pid.append(nbrs_array[current])

            # check if particle overlaps with the global domain
            if (center[0] - r[i] < xmax) and (center[0] + r[i] > xmin) and\
               (center[1] - r[i] < ymax) and (center[1] + r[i] > ymin):
                   indicies.append(i)

        # extract ghost particles
        send_particles = particles.extract(buffer_ids)

        for i in range(indices.length):

            pid = indices.data[i]

            # bounding box of particle
            for ii in range(3):
                for jj in range(3):

                    # skip original position
                    if ii == jj == 1: continue

                    # shifted periodic coordinates
                    center[0] = x[0][pid] + (ii-1)*self.domain.xtranslate
                    center[1] = x[1][pid] + (jj-1)*self.domain.ytranslate

                    # check if particle overlaps with the global domain
                    if (center[0] - r[pid] < xmax) and (center[0] + r[pid] > xmin) and\
                       (center[1] - r[pid] < ymax) and (center[1] + r[pid] > ymin):

                # find overlaping processors
                nbrs_array.reset()
                num_neighbors = self.load_balance.global_tree._get_nearest_neighbors(
                        center, r[pid], self.load_balance.leaf_proc, self.rank, nbrs)

                if num_neighbors != 0:

                    nbrs_array = nbrs.get_npy_array()

                    # put in processors in order to avoid duplicates
                    nbrs_array.sort()

                    # put particles in buffer
                    for current range(num_neighbors):

                        if current == num_neighbors-1:
                            if nbrs_array[current] != nbrs_array[current-1]:
                                global_buffer_ids.append(i)
                                global_buffer_pid.append(nbrs_array[current])
                                xg.append(center[0])
                                yg.append(center[1])
                                num_global_export += 1
                        else:
                            if nbrs_array[current] != nbrs_array[current+1]:
                                global_buffer_ids.append(i)
                                global_buffer_pid.append(nbrs_array[current])
                                xg.append(center[0])
                                yg.append(center[1])
                                num_global_export += 1

        if num_global_export != 0:

            global_particles = pc.extract(global_buffer_ids)
            global_particles["position-x"][:] = xg_npy
            global_particles["position-y"][:] = yg_npy
            global_particles["tag"][:] = ParticleTAGS.ExteriorGhost

            send_particles.append(global_particles)
            buffer_ids.append(global_buffer_ids)
            buffer_pids.append(global_buffer_pids)

        # count the number of particles going to each processor
        send_particles[:] = np.bincount(buffer_pids, minlength=size)

        # recieve the number of particles from each processor
        comm.Alltoall(sendbuf=send_particles, recbuf=recv_particles)

        # resize arrays for incoming particles
        local_num = pc.get_number_of_particles()
        pc.extend(np.sum(recv_particlesf))

        exchange_particles(pc, send_data, send_particles, recv_particles, local_num, comm)
