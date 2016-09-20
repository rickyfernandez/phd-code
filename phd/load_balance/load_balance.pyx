import numpy as np
from mpi4py import MPI
from ..utils.particle_tags import ParticleTAGS
from ..utils.exchange_particles import exchange_particles

from .tree cimport Node
from ..domain.domain cimport DomainLimits
from ..hilbert.hilbert cimport hilbert_key_2d, hilbert_key_3d


cdef class LoadBalance:
    def __init__(self, DomainLimits domain, object comm=None,
            np.float64_t factor=0.1, int min_in_leaf=32, np.int32_t order=21):
        """Constructor for load balance

        Parameters
        ----------
        domain : DomainLimits
            Domain coordinate information.
        comm : MPI.COMM_WORLD
            MPI communicator.
        order : int
            The number of bits per dimension for constructing hilbert keys.
        """
        cdef int i

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.order = order
        self.factor = factor

        fudge_factor = 1.001
        self.box_length = domain.max_length*fudge_factor
        self.fac = (1 << self.order) / self.box_length

        self.corner = np.zeros(3, dtype=np.float64)
        for i in range(domain.dim):
            self.corner[i] = (domain.bounds[0][i] + domain.bounds[1][i])*0.5 - 0.5*self.box_length

        self.leaf_pid = LongArray()
        self.export_ids = LongArray()
        self.export_pid = LongArray()

        if domain.dim == 2:
            self.hilbert_func = hilbert_key_2d
        elif domain.dim == 3:
            self.hilbert_func = hilbert_key_3d
        else:
            raise RuntimeError("Wrong dimension for tree")

        self.tree = Tree(self.corner, self.box_length, domain.dim,
                factor, min_in_leaf, order)

    def decomposition(self, ParticleContainer pc):
        """Perform domain decomposition
        """
        cdef np.ndarray ind
        cdef sendbuf, recvbuf
        cdef np.ndarray export_ids_npy, export_pid_npy
        cdef np.ndarray local_work, global_work

        cdef int temp

        # remove current (if any) ghost particles
        pc.remove_tagged_particles(ParticleTAGS.Ghost)

        # generate hilbert keys for real particles and create
        # global tree over all process
        self._compute_hilbert_keys(pc)
        self.tree.construct_global_tree(pc, self.comm)

        local_work  = np.zeros(self.tree.number_leaves, dtype=np.int32)
        global_work = np.zeros(self.tree.number_leaves, dtype=np.int32)
        self._calculate_local_work(pc, local_work)

        # gather work across all processors
        self.comm.Allreduce(sendbuf=local_work, recvbuf=global_work, op=MPI.SUM)
        self._find_split_in_work(global_work)

        # collect particle for export
        self.export_ids.reset()
        self.export_pid.reset()
        self._collect_particles_export(pc, self.export_ids, self.export_pid,
                self.leaf_pid, self.rank)

        # arrange particles in process order
        export_ids_npy = self.export_ids.get_npy_array()
        export_pid_npy = self.export_pid.get_npy_array()

        ind = export_pid_npy.argsort()
        export_ids_npy[:] = export_ids_npy[ind]
        export_pid_npy[:] = export_pid_npy[ind]

        # count number of particles to send to each process
        recvbuf = np.zeros(self.size, dtype=np.int32)
        sendbuf = np.bincount(export_pid_npy,
                minlength=self.size).astype(np.int32)

        # how many particles are being sent from each process
        self.comm.Alltoall(sendbuf=sendbuf, recvbuf=recvbuf)

        # extract particles to send 
        send_data = pc.get_sendbufs(export_ids_npy)
        pc.remove_items(export_ids_npy)
        temp = pc.get_number_of_particles()

        # exchange load balance particles
        pc.extend(np.sum(recvbuf))
        exchange_particles(pc, send_data, sendbuf, recvbuf,
                temp, self.comm)

        pc.align_particles()
        pc['process'][:] = self.rank

    cdef void _calculate_local_work(self, ParticleContainer pc, np.ndarray work):
        """Calculate global work by calculating local work in each leaf. Then sum
        work across all process. Currently the work is just the the number of
        particles in each leaf.
        """
        cdef int i
        cdef Node* node
        cdef LongLongArray keys = pc.get_carray("key")

        # work is the number of local particles in leaf
        for i in range(pc.get_number_of_particles()):
            node = self.tree.find_leaf(keys.data[i])
            work[node.array_index] += 1

    cdef void _find_split_in_work(self, np.ndarray global_work):
        """Partition the global leaves amongst the process such that each process
        has roughly equal work load.
        """
        cdef int i, j, cum_sum
        cdef int total_work, part_per_proc

        self.leaf_pid.resize(global_work.size)

        total_work = global_work.sum()
        if total_work%self.size == 0:
            part_per_proc = total_work/self.size
        else:
            part_per_proc = total_work/self.size + 1

        j = 1
        cum_sum = 0
        for i in range(global_work.size):
            cum_sum += global_work[i]
            if cum_sum > j*part_per_proc:
                j += 1
            self.leaf_pid.data[i] = j - 1

    cdef void _collect_particles_export(self, ParticleContainer pc, LongArray part_ids,
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
        cdef LongLongArray keys = pc.get_carray("key")

        for i in range(pc.get_number_of_particles()):

            node = self.tree.find_leaf(keys.data[i])
            pid  = leaf_pid.data[node.array_index]

            if pid != my_pid:
                part_ids.append(i)
                part_pid.append(pid)

    cdef void _compute_hilbert_keys(self, ParticleContainer pc):
        """Compute hilbert key for each particle. The container is assumed to only have
        real particles.

        Parameters
        ----------
        pc : ParticleContainer
            Particle container holding all information of particles in the simulation.
        """
        cdef int i, j
        cdef Node* node
        cdef np.int32_t xh[3]
        cdef np.float64_t *x[3]
        cdef LongLongArray keys = pc.get_carray("key")

        pc.pointer_groups(x, pc.named_groups['position'])

        for i in range(pc.get_number_of_particles()):
            for j in range(self.tree.dim):
                xh[j] = <np.int32_t> ( (x[j][i] - self.corner[j])*self.fac )

            keys.data[i] = self.hilbert_func(xh[0], xh[1], xh[2], self.order)
