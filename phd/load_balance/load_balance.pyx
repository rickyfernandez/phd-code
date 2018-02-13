import phd
import numpy as np
from mpi4py import MPI

from ..utils.particle_tags import ParticleTAGS
from ..utils.exchange_particles import exchange_particles

from .tree cimport Node
from ..domain.domain_manager cimport DomainManager
from ..hilbert.hilbert cimport hilbert_key_2d, hilbert_key_3d


cdef class LoadBalance:
    def __init__(self, np.float64_t factor=0.1, int min_in_leaf=32, np.int32_t order=21, **kwargs):
        """Constructor for load balance

        Parameters
        ----------
        order : int
            The number of bits per dimension for constructing hilbert keys.
        """
        self.order = order
        self.factor = factor
        self.min_in_leaf = min_in_leaf

        self.domain_info_added = False

        self.leaf_pid = LongArray()
        self.export_ids = LongArray()
        self.export_pid = LongArray()

    def add_domain_info(self, DomainManager domain_manager):
        cdef int i, k

        for i in range(2):
            for k in range(domain_manager.dim):
                self.bounds[i][k] = domain_manager.bounds[i][k]

        self.dim = domain_manager.dim
        self.box_length = domain_manager.max_length
        self.domain_info_added = True

    def initialize(self):
        """Constructor for load balance
        """
        cdef int i
        cdef np.ndarray corner = np.zeros(3)

        if not self.domain_info_added:
            raise RuntimeError("ERROR: Domain information not added")
        self.fac = 2**self.order / self.box_length

        for i in range(self.dim):
            corner[i] = self.bounds[0][i]

        if self.dim == 2:
            self.hilbert_func = hilbert_key_2d
        elif self.dim == 3:
            self.hilbert_func = hilbert_key_3d
        else:
            raise RuntimeError("Wrong dimension for tree")

        self.tree = Tree(corner, self.box_length, self.dim,
                self.factor, self.min_in_leaf, self.order)

    def decomposition(self, CarrayContainer particles):
        """Perform domain decomposition
        """
        cdef np.ndarray ind
        cdef sendbuf, recvbuf
        cdef np.ndarray export_ids_npy, export_pid_npy
        cdef np.ndarray local_work, global_work

        cdef int temp

        # remove current (if any) ghost particles
        particles.remove_tagged_particles(ParticleTAGS.Ghost)

        # generate hilbert keys for real particles and create
        # global tree over all process
        self.compute_hilbert_keys(particles)
        self.tree.construct_global_tree(particles, phd._comm)

        local_work  = np.zeros(self.tree.number_leaves, dtype=np.int32)
        global_work = np.zeros(self.tree.number_leaves, dtype=np.int32)
        self.calculate_local_work(particles, local_work)

        # gather work across all processors
        phd._comm.Allreduce(sendbuf=local_work, recvbuf=global_work, op=MPI.SUM)
        self.find_split_in_work(global_work)

        # collect particle for export
        self.export_ids.reset()
        self.export_pid.reset()
        self.collect_particles_export(particles, self.export_ids, self.export_pid,
                self.leaf_pid, phd._rank)

        # arrange particles in process order
        export_ids_npy = self.export_ids.get_npy_array()
        export_pid_npy = self.export_pid.get_npy_array()

        ind = export_pid_npy.argsort()
        export_ids_npy[:] = export_ids_npy[ind]
        export_pid_npy[:] = export_pid_npy[ind]

        # count number of particles to send to each process
        recvbuf = np.zeros(phd._size, dtype=np.int32)
        sendbuf = np.bincount(export_pid_npy,
                minlength=phd._size).astype(np.int32)

        # how many particles are being sent from each process
        phd._comm.Alltoall(sendbuf=sendbuf, recvbuf=recvbuf)

        # extract particles to send 
        send_data = particles.get_sendbufs(export_ids_npy)
        particles.remove_items(export_ids_npy)
        temp = particles.get_carray_size()

        # exchange load balance particles
        particles.extend(np.sum(recvbuf))
        exchange_particles(particles, send_data, sendbuf, recvbuf,
                temp, phd._comm)

        particles["process"][:] = phd._rank

    cdef void calculate_local_work(self, CarrayContainer particles, np.ndarray work):
        """Calculate global work by calculating local work in each leaf. Then sum
        work across all process. Currently the work is just the the number of
        particles in each leaf.
        """
        cdef int i
        cdef Node* node
        cdef LongLongArray keys = particles.get_carray("key")

        # work is the number of local particles in leaf
        for i in range(particles.get_carray_size()):
            node = self.tree.find_leaf(keys.data[i])
            work[node.array_index] += 1

    cdef void find_split_in_work(self, np.ndarray global_work):
        """Partition the global leaves amongst the process such that each process
        has roughly equal work load.
        """
        cdef int i, j, cum_sum
        cdef int total_work, part_per_proc

        self.leaf_pid.resize(global_work.size)

        total_work = global_work.sum()
        if total_work%phd._size == 0:
            part_per_proc = total_work/phd._size
        else:
            part_per_proc = total_work/phd._size + 1

        j = 1
        cum_sum = 0
        for i in range(global_work.size):
            cum_sum += global_work[i]
            if cum_sum > j*part_per_proc:
                j += 1
            self.leaf_pid.data[i] = j - 1

    cdef void collect_particles_export(self, CarrayContainer particles, LongArray part_ids,
        LongArray part_pid, LongArray leaf_pid, int my_pid):
        """
        Collect export particle indices and the process that it will be sent too.

        Parameters
        ----------
        keys : ndarray
            Hilbert key for each real particle.
        part_ids : ndarray
            Particle indices that need to be exported.
        proc_ids : ndarray
            The process that each exported particle must be sent too.
        leaf_procs : ndarray
            Rank of process for each leaf.
        my_proc : int
            Rank of current process.

        """
        cdef Node *node
        cdef int i, pid
        cdef LongLongArray keys = particles.get_carray("key")

        for i in range(particles.get_carray_size()):

            node = self.tree.find_leaf(keys.data[i])
            pid  = leaf_pid.data[node.array_index]

            if pid != my_pid:
                part_ids.append(i)
                part_pid.append(pid)

    cdef void compute_hilbert_keys(self, CarrayContainer particles):
        """Compute hilbert key for each particle. The container is assumed to only have
        real particles.

        Parameters
        ----------
        particles : CarrayContainer
            Particle container holding all information of particles in the simulation.
        """
        cdef int i, j
        cdef Node* node
        cdef np.int32_t xh[3]
        cdef np.float64_t *x[3]
        cdef LongLongArray keys = particles.get_carray("key")

        particles.pointer_groups(x, particles.carray_named_groups["position"])

        for i in range(particles.get_carray_size()):
            for j in range(self.tree.dim):
                xh[j] = <np.int32_t> ( (x[j][i] - self.tree.domain_corner[j])*self.fac )

            keys.data[i] = self.hilbert_func(xh[0], xh[1], xh[2], self.order)
