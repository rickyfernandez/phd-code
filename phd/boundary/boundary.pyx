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
cdef int ExteriorGhost= ParticleTAGS.ExteriorGhost

cdef _reflective(ParticleContainer pc, DomainLimits domain):

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

    cdef dim = domain.dim

    pc.extract_field_vec_ptr(x, "position")
    pc.extract_field_vec_ptr(v, "velocity")

    for i in range(pc.get_number_of_particles()):

        for j in range(dim):

            # create bounding box for particle
            box[0][j] = x[j][i] - r.data[i]
            box[1][j] = x[j][i] + r.data[i]

        # lower boundary
        for j in range(dim):

            # test if ghost particle should be created
            if box[0][j] < domain.bounds[0][j]:

                # copy particle information
                for k in range(dim):
                    xp[k] = x[k][i]
                    vp[k] = v[k][i]

                # reflect particle position and velocity 
                xp[j] =  x[j][i] - 2*(x[j][i] - domain.bounds[0][j])
                vp[j] = -v[j][i]

                # store new ghost particle
                ghost_particle.push_back(Particle(xp, vp, self.dim))
                indices.append(i)

        # upper boundary
        for j in range(dim):

            # test if ghost particle should be created
            if domain.bounds[1][j] < box[1][j]:

                # copy particle information
                for k in range(dim):
                    xp[k] = x[k][i]
                    vp[k] = v[k][i]

                # reflect particle position and velocity
                xp[j] =  x[j][i] - 2*(x[j][i] - domain.bounds[1][j])
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
        for j in range(dim):
            xg[j][i] = p.x[j]
            vg[j][i] = p.v[j]

    # add reflect ghost to particle container
    pc.append_container(exterior_ghost)

cdef _periodic(ParticleContainer pc, DomainLimits domain):

    cdef CarrayContainer exterior_ghost

    cdef LongArray maps = pc.get_carray("map")
    cdef DoubleArray r = pc.get_carray("radius")

    cdef IntArray tags
    cdef IntArray types

    cdef vector[Particle] ghost_particle
    cdef Particle *p

    cdef np.float64_t box[2][3], xp[3], vp[3]
    cdef np.float64_t *x[3]
    cdef np.float64_t *xg[3]

    cdef LongArray indices = LongArray()
    cdef int i, j, m, num_ghost = 0

    cdef dim = domain.dim

    pc.extract_field_vec_ptr(x, "position")

    for j in range(dim):
        vg[j] = 0

    for i in range(pc.get_number_of_particles()):

        # check if particle intersects global domain box
        for j in range(dim):
            if x[j][i] + r.data[i] < domain.bounds[0][j]: continue
            if x[j][i] - r.data[i] < domain.bounds[1][j]: continue

        # shift particle to create ghost particle
        for m in range(3**dim):

            # create shift indices
            ii = m%3; jj = m/3; kk = m/9

            xp[0] = x[0][i] + (ii-1)*domain.translate[0]
            xp[1] = x[1][i] + (jj-1)*domain.translate[1]
            xp[2] = x[2][i] + (kk-1)*domain.translate[2]

            store = 1
            for j in range(dim):
                if x[j][i] + r.data[i] < domain.bounds[0][j]: store = 0
                if x[j][i] - r.data[i] < domain.bounds[1][j]: store = 0

            # store new ghost particle
            if store:
                ghost_particle.push_back(Particle(xp, vp, self.dim))
                indices.append(i)

    exterior_ghost = pc.extract_items(indices.get_npy_array())
    exterior_ghost.extract_field_vec_ptr(xg, "position")

    tags = exterior_ghost.get_carray("tag")
    types = exterior_ghost.get_carray("type")

    # transfer data to ghost particles
    for i in range(exterior_ghost.get_number_of_items()):

        maps.data[i] = indices.data[i]
        tags.data[i] = ParticleTAGS.Ghost
        types.data[i] = ParticleTAGS.Exterior

        p = &ghost_particle[i]
        for j in range(dim):
            xg[j][i] = p.x[j]

    # add periodic ghost to particle container
    pc.append_container(exterior_ghost)

cdef _periodic_parallel(self, ParticleContainer pc, LongArray buffer_ids, LongArray buffer_pid):

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

        # check if particle intersects global domain box
        for j in range(dim):
            if x[j][i] + r.data[i] < domain.bounds[0][j]: continue
            if x[j][i] - r.data[i] < domain.bounds[1][j]: continue

        # shift particle to create ghost particle
        for m in range(3**dim):

            # create shift indices
            ii = m%3; jj = m/3; kk = m/9

            center[0] = x[0][i] + (ii-1)*domain.translate[0]
            center[1] = x[1][i] + (jj-1)*domain.translate[1]
            center[2] = x[2][i] + (kk-1)*domain.translate[2]

            store = 1
            for j in range(dim):
                if center[j][i] + r.data[i] < domain.bounds[0][j]: store = 0
                if center[j][i] - r.data[i] < domain.bounds[1][j]: store = 0

            if store == 1:

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

cdef class Boundary:
    def __init__(self, DomainLimits domain, int boundary_type):

        self.domain = domain

        if boundary_type == 0: # reflective
            self._create_exterior_ghost_particles_func = _reflective
        if boundary_type == 1: # periodic -- not implemented yet
            self._create_exterior_ghost_particles_func = _periodic

   cdef _update_exterior_ghost_particles(self, ParticleContainer pc):

        cdef IntArray types  = pc.get_carray("type")
        cdef LongArray maps  = pc.get_carray("map")
        cdef DoubleArray vol = pc.get_carray("volume")

        cdef np.float64_t *x[3], *dcomx[3]
        cdef int i, j, dim = self.domain.dim

        pc.extract_field_vec_ptr(x, "position")
        pc.extract_field_vec_ptr(dcomx, "dcom")

        for i in range(pc.get_number_of_particles()):
            if types.data[i] == ExteriorGhost:

                j = maps.data[i]

                vol.data[i] = vol.data[j]
                for k in range(dim):
                    dcomx[k][i] = dcomx[k][j]

    cdef int _create_ghost_particles(self, ParticleContainer pc):
        return self._create_exterior_ghost_particles_func(pc)

cdef class BoundaryParallel(Boundary):

    def __init__(self, DomainLimits domain, int boundary_type, object load_bal, object comm):

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.load_bal = load_bal

        self.buffer_ids = LongArray()
        self.buffer_pid = LongArray()

        self.domain = domain

        if boundary_type == 0: # reflective
            self._create_exterior_ghost_particles_func = _reflective
        if boundary_type == 1: # periodic
            self._create_exterior_ghost_particles_func = _periodic_parallel

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

        self._create_interior_ghost_particles(pc)
        self._create_exterior_ghost_particles_func(pc)
        self._send_ghost_particles(pc)

        return pc.get_number_of_particles() - num_local
