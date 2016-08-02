import numpy as np
cimport numpy as np

from boundary cimport Particle
from libcpp.vector cimport vector
from domain.domain cimport DomainLimits
from utils.particle_tags import ParticleTAGS
from load_balance.tree cimport Node, BaseTree
from utils.exchange_particles import exchange_particles
from utils.carray cimport DoubleArray, LongLongArray, LongArray, IntArray
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

cdef class BoundaryParallelBase:

    def __init__(self, DomainLimits domain, object load_bal, object comm):

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.load_bal = load_bal
        self.dim = 2

        self.buffer_ids = LongArray()
        self.buffer_pid = LongArray()

        self.bounds[0][0] = domain.xmin
        self.bounds[1][0] = domain.xmax

        self.bounds[0][1] = domain.ymin
        self.bounds[1][1] = domain.ymax

    cdef _interior_ghost_particles(self, ParticleContainer pc):

        cdef LongLongArray keys = pc.get_carray("key")
        cdef DoubleArray r = pc.get_carray("radius")

        cdef Node* node

        cdef np.float64_t* x[3]
        cdef double center[3]

        cdef IntArray types = pc.get_carray("type")
        cdef LongArray nbrs = LongArray()
        cdef np.ndarray nbrs_npy, leaf_npy

        cdef BaseTree glb_tree = self.load_bal.global_tree
        cdef int i, j, num_neighbors, dim

        # clear existing buffers
        self.buffer_ids.reset()
        self.buffer_pid.reset()

        # problem dimension
        dim = self.load_bal.dim
        leaf_npy = self.load_bal.leaf_proc

        pc.extract_field_vec_ptr(x, "position")

        # loop over local particles - there should be no ghost
        for i in range(pc.get_number_of_particles()):

            # find size of leaf where particle lives in global tree
            node = glb_tree._find_leaf(keys.data[i])
            r.data[i] = min(0.5*node.box_length/glb_tree.domain_fac, r.data[i])

            # center of bounding box of particle
            for j in range(dim):
                center[j] = x[j][i]

            # find overlaping processors
            nbrs.reset()
            num_neighbors = glb_tree._get_nearest_process_neighbors(
                    center, r.data[i], leaf_npy, self.rank, nbrs)
            nbrs_npy = nbrs.get_npy_array()

            if num_neighbors != 0:

                # put in processors in order to avoid duplicates
                nbrs_npy.sort()

                # put particles in buffer
                self.buffer_ids.append(i)            # store particle id
                self.buffer_pid.append(nbrs_npy[0])  # store export processor

                for j in range(1, num_neighbors):

                    if nbrs_npy[j] != nbrs_npy[j-1]:
                        self.buffer_ids.append(i)            # store particle id
                        self.buffer_pid.append(nbrs_npy[j])  # store export processor

    cdef _exterior_ghost_particles(self, ParticleContainer pc):
        msg = "BoundaryParallelBase::_exterior_ghost_particles called!"
        raise NotImplementedError(msg)

    cdef _send_ghost_particles(self, ParticleContainer pc):

        cdef CarrayContainer ghost_particles
        cdef np.ndarray ind, send_particles, recv_particles, buf_pid, buf_ids
        cdef int num_new_ghost

        # use numpy arrays for convience
        buf_pid = self.buffer_pid.get_npy_array()
        buf_ids = self.buffer_ids.get_npy_array()

        # organize particle processors
        ind = np.argsort(buf_pid)
        buf_ids[:] = buf_ids[ind]
        buf_pid[:] = buf_pid[ind]

        # bin the number of particles being sent to each processor
        send_particles = np.bincount(buf_pid,
                minlength=self.size).astype(np.int32)

        # how many particles are you receiving from each processor
        recv_particles = np.empty(self.size, dtype=np.int32)
        self.comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)
        num_new_ghost = np.sum(recv_particles)

        # extract particles that will be removed
        ghost_particles = pc.extract_items(buf_ids)
        ghost_particles["tag"][:] = ParticleTAGS.Ghost

        # make room for new ghost particles
        displacement = pc.get_number_of_particles()
        pc.extend(num_new_ghost)

        # do the excahnge
        exchange_particles(pc, ghost_particles, send_particles, recv_particles,
                displacement, self.comm)

    cdef int _create_ghost_particles(self, ParticleContainer pc):
        cdef int num_local = pc.get_number_of_particles()

        self._interior_ghost_particles(pc)
        self._exterior_ghost_particles(pc)
        self._send_ghost_particles(pc)

        return pc.get_number_of_particles() - num_local

cdef class ReflectParallel(BoundaryParallelBase):

    cdef _exterior_ghost_particles(self, ParticleContainer pc):

        cdef CarrayContainer exterior_ghost

        cdef LongArray maps = pc.get_carray("map")
        cdef DoubleArray r = pc.get_carray("radius")

        cdef IntArray tags
        cdef IntArray types

        cdef vector[Particle] ghost_particle
        cdef Particle *p

        cdef np.float64_t box[2][3], xp[3], vp[3]
        cdef np.float64_t *x[3], *v[3]
        cdef np.float64_t *xg[3], *vg[3]

        cdef LongArray indices = LongArray()
        cdef int i, num_ghost = 0

        pc.extract_field_vec_ptr(x, "position")
        pc.extract_field_vec_ptr(v, "velocity")

        for i in range(pc.get_number_of_particles()):

            for j in range(self.dim):

                # create bounding box for particle
                box[0][j] = x[j][i] - r.data[i]
                box[1][j] = x[j][i] + r.data[i]

            # lower boundary
            for j in range(self.dim):

                # test if ghost particle should be created
                if box[0][j] < self.bounds[0][j]:

                    # copy particle information
                    for k in range(self.dim):
                        xp[k] = x[k][i]
                        vp[k] = v[k][i]

                    # reflect particle position and velocity 
                    xp[j] =  x[j][i] - 2*(x[j][i] - self.bounds[0][j])
                    vp[j] = -v[j][i]

                    # store new ghost particle
                    ghost_particle.push_back(Particle(xp, vp, self.dim))
                    indices.append(i)

            # upper boundary
            for j in range(self.dim):

                # test if ghost particle should be created
                if self.bounds[1][j] < box[1][j]:

                    # copy particle information
                    for k in range(self.dim):
                        xp[k] = x[k][i]
                        vp[k] = v[k][i]

                    # reflect particle position and velocity
                    xp[j] =  x[j][i] - 2*(x[j][i] - self.bounds[1][j])
                    vp[j] = -v[j][i]

                    # store new ghost particle
                    ghost_particle.push_back(Particle(xp, vp, self.dim))
                    indices.append(i)

        exterior_ghost = pc.extract_items(indices.get_npy_array())
        exterior_ghost.extract_field_vec_ptr(xg, "position")
        exterior_ghost.extract_field_vec_ptr(vg, "velocity")

        tags = exterior_ghost.get_carray("tag")
        types = exterior_ghost.get_carray("type")

        # transfer data to ghost particles
        for i in range(exterior_ghost.get_number_of_items()):

            maps.data[i] = indices.data[i]
            tags.data[i] = ParticleTAGS.Ghost
            types.data[i] = ParticleTAGS.Exterior

            p = &ghost_particle[i]
            for j in range(self.dim):
                xg[j][i] = p.x[j]
                vg[j][i] = p.v[j]

        # add reflect ghost to particle container
        pc.append_container(exterior_ghost)


#cdef class PeriodicParallel(BoundaryParallelBase):
#
#    cdef int _create_ghost_particles(self, ParticleContainer pc):
#
#        cdef LongLongArray keys = pc.get_carray("key")
#
#        cdef DoubleArray xg = DoubleArray()
#        cdef DoubleArray yg = DoubleArray()
#        cdef DoubleArray zg = DoubleArray()
#
#        cdef np.float64_t* x[3]
#        cdef int i, j, current, num_neighbors
#
#        self.buffer_ids.reset()
#        self.buffer_pid.reset()
#        self.num_export = 0
#
#        pc.extract_field_vec_ptr(x, "position")
#
#        for i in range(pc.get_number_of_particles()):
#
#            # first we do interior ghost particles
#
#            # find the size of the leaf where particle lives in
#            node = self.load_balance.global_tree._find_leaf(keys.data[i])
#            r[i] = min(0.25*node.box_length, r[i])
#
#            # bounding box of particle
#            for j in range(self.dim):
#                center[j] = x[i][j]
#
#            # find overlaping processors
#            nbrs_array.reset()
#            num_neighbors = self.load_balance.global_tree._get_nearest_neighbors(
#                    center, r[i], self.load_balance.leaf_proc, self.rank, nbrs)
#            nbrs_array = nbrs.get_npy_array()
#
#            # put in processors in order to avoid duplicates
#            nbrs_array.sort()
#
#            # put particles in buffer
#            for current range(num_neighbors):
#
#                if current == num_neighbors-1:
#                    if nbrs_array[current] != nbrs_array[current-1]:
#                        buffer_ids.append(i)
#                        buffer_pid.append(nbrs_array[current])
#                        num_export += 1
#                else:
#                    if nbrs_array[current] != nbrs_array[current+1]:
#                        buffer_ids.append(i)
#                        buffer_pid.append(nbrs_array[current])
#
#            # check if particle overlaps with the global domain
#            if (center[0] - r[i] < xmax) and (center[0] + r[i] > xmin) and\
#               (center[1] - r[i] < ymax) and (center[1] + r[i] > ymin):
#                   indices.append(i)
#
#        # extract ghost particles
#        send_particles = particles.extract(buffer_ids)
#
#        for i in range(indices.length):
#
#            pid = indices.data[i]
#
#            # bounding box of particle
#            for ii in range(3):
#                for jj in range(3):
#
#                    # skip original position
#                    if ii == jj == 1: continue
#
#                    # shifted periodic coordinates
#                    center[0] = x[0][pid] + (ii-1)*self.domain.xtranslate
#                    center[1] = x[1][pid] + (jj-1)*self.domain.ytranslate
#
#                    # check if particle overlaps with the global domain
#                    if (center[0] - r[pid] < xmax) and (center[0] + r[pid] > xmin) and\
#                       (center[1] - r[pid] < ymax) and (center[1] + r[pid] > ymin):
#
#                # find overlaping processors
#                nbrs_array.reset()
#                num_neighbors = self.load_balance.global_tree._get_nearest_neighbors(
#                        center, r[pid], self.load_balance.leaf_proc, self.rank, nbrs)
#
#                if num_neighbors != 0:
#
#                    nbrs_array = nbrs.get_npy_array()
#
#                    # put in processors in order to avoid duplicates
#                    nbrs_array.sort()
#
#                    # put particles in buffer
#                    for current range(num_neighbors):
#
#                        if current == num_neighbors-1:
#                            if nbrs_array[current] != nbrs_array[current-1]:
#                                global_buffer_ids.append(i)
#                                global_buffer_pid.append(nbrs_array[current])
#                                xg.append(center[0])
#                                yg.append(center[1])
#                                num_global_export += 1
#                        else:
#                            if nbrs_array[current] != nbrs_array[current+1]:
#                                global_buffer_ids.append(i)
#                                global_buffer_pid.append(nbrs_array[current])
#                                xg.append(center[0])
#                                yg.append(center[1])
#                                num_global_export += 1
#
#        if num_global_export != 0:
#
#            global_particles = pc.extract(global_buffer_ids)
#            global_particles["position-x"][:] = xg_npy
#            global_particles["position-y"][:] = yg_npy
#            global_particles["tag"][:] = ParticleTAGS.ExteriorGhost
#
#            send_particles.append(global_particles)
#            buffer_ids.append(global_buffer_ids)
#            buffer_pids.append(global_buffer_pids)
#
#        # count the number of particles going to each processor
#        send_particles[:] = np.bincount(buffer_pids, minlength=size)
#
#        # recieve the number of particles from each processor
#        comm.Alltoall(sendbuf=send_particles, recbuf=recv_particles)
#
#        # resize arrays for incoming particles
#        local_num = pc.get_number_of_particles()
#        pc.extend(np.sum(recv_particles))
#
#        exchange_particles(pc, send_data, send_particles, recv_particles, local_num, comm)
