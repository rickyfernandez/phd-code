import itertools
import numpy as np
from mpi4py import MPI

from .tree import QuadTree
from utils.particle_tags import ParticleTAGS
from hilbert.hilbert import hilbert_key_2d
from utils.exchange_particles import exchange_particles


class LoadBalance(object):

    def __init__(self, particles, domain, comm=None, factor=0.1, order=21):
        """Constructor for load balance

        Parameters
        ----------
        particles : ParticleArray
            Particle container of fluid values.
        corner : ndarray
            Left corner of the box simulation.
        box_length : double
            Size of the box simulation.
        comm : MPI.COMM_WORLD
            MPI communicator.
        order : int
            The number of bits per dimension for constructing
                hilbert keys
        """
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.factor = factor

        self.order = order
        self.number_real_particles = particles.num_real_particles
        self.particles = particles

        fudge_factor = 1.001
        self.box_length = max(domain.xtranslate,
                domain.ytranslate)*fudge_factor

        self.corner = np.array([
            domain.xtranslate*0.5-0.5*self.box_length,
            domain.ytranslate*0.5-0.5*self.box_length],
            dtype=np.float64)

        self.keys = None
        self.sorted_keys = None
        self.global_num_real_particles = None

        self.leaf_proc = None
        self.global_tree = None

    def calculate_hilbert_keys(self):
        """map particle positions to hilbert space"""
        num_real_part = self.particles.num_real_particles

        x = self.particles['position-x']
        y = self.particles['position-y']

        # normalize coordinates to hilbert space
        fac = (1 << self.order)/ self.box_length

        # create hilbert key for each particle
        # ghost particles should have been discarded
        self.keys = np.array([hilbert_key_2d(
            np.int64((x[i]-self.corner[0])*fac),
            np.int64((y[i]-self.corner[1])*fac), self.order) for i in range(num_real_part)],
            dtype = np.int64)

        self.sorted_keys = np.sort(self.keys)

        # copy hilbert keys to particles
        self.particles["key"][:] = self.keys


    def decomposition(self):
        """Perform domain decomposition
        """

        # remove current (if any) ghost particles
        self.particles.remove_tagged_particles(ParticleTAGS.Ghost)

        # generate hilbert keys for real particles and create
        # global tree over all process
        self.calculate_hilbert_keys()
        global_tree = self.build_global_tree()

        # partion tree leaves by the amount work done
        # in each leaf
        self.calculate_global_work(global_tree)
        self.find_split_in_work()

        # count the number of particles that need to be exported
        num_export = global_tree.count_particles_export(self.keys, self.leaf_proc, self.rank)

        self.export_ids = np.empty(num_export, dtype=np.int32)
        self.export_proc = np.empty(num_export, dtype=np.int32)

        # collect particles to be exported with their process location
        global_tree.collect_particles_export(self.keys, self.export_ids, self.export_proc,
                self.leaf_proc, self.rank)

        # arrange particles in process order
        ind = self.export_proc.argsort()
        self.export_proc = self.export_proc[ind]
        self.export_ids  = self.export_ids[ind]
        #send_data = self.particles.get_sendbufs(self.export_ids)

        # count the number of particles to send to each process
        send_particles = np.bincount(self.export_proc,
                minlength=self.size).astype(np.int32)

        # extract particles to send 
        send_data = self.particles.get_sendbufs(self.export_ids)

        # remove exported particles
        self.particles.remove_items(self.export_ids)

        # how many particles are being sent from each process
        recv_particles = np.empty(self.size, dtype=np.int32)
        self.comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)

        # resize particle array and place incoming particles at the
        # end of the array
        #displacement = self.particles.num_real_particles
        displacement = self.particles.get_number_of_particles()
        num_incoming_particles = np.sum(recv_particles)
        self.particles.extend(num_incoming_particles)

        # exchange load balance particles
        exchange_particles(self.particles, send_data, send_particles, recv_particles,
                displacement, self.comm)

        # align particles and count the number of real particles
        self.particles.align_particles()

        # label all new particles to current process
        self.particles['process'][:] = self.rank

        self.global_tree = global_tree

    def build_global_tree(self):
        """Build a global tree on all process. This algorithm follows springel (2005). First
        create local tree from local particles by subdividing hilbert keys. Then all local
        leaves are collected to all process and a second tree is created using the leaves
        (i.e. the hilbert cuts).
        """
        # collect number of real particles from all process
        sendbuf = np.array([self.number_real_particles], dtype=np.int32)
        proc_num_real_particles = np.empty(self.size, dtype=np.int32)

        self.comm.Allgather(sendbuf=sendbuf, recvbuf=proc_num_real_particles)
        self.global_num_real_particles = np.sum(proc_num_real_particles)

        # construct local tree
        local_tree = QuadTree(self.global_num_real_particles, self.sorted_keys,
                self.corner, self.box_length,
                total_num_process=self.size, factor=self.factor, order=self.order)
        local_tree.build_tree()

        # collect leaf start keys and number of particles for each leaf and send to all process
        leaf_keys, num_part_leaf = local_tree.collect_leaves_for_export()

        # prepare to bring all leaves form all local trees
        counts = np.empty(self.size, dtype=np.int32)
        sendbuf = np.array([leaf_keys.size], dtype=np.int32)
        self.comm.Allgather(sendbuf=sendbuf, recvbuf=counts)

        global_num_leaves = np.sum(counts)
        offsets = np.zeros(self.size)
        offsets[1:] = np.cumsum(counts)[:-1]

        global_leaf_keys = np.empty(global_num_leaves, dtype=np.int64)
        global_num_part_leaves = np.empty(global_num_leaves, dtype=np.int32)

        self.comm.Allgatherv(leaf_keys, [global_leaf_keys, counts, offsets, MPI.INT64_T])
        self.comm.Allgatherv(num_part_leaf, [global_num_part_leaves, counts, offsets, MPI.INT])

        # sort global leaves
        ind = global_leaf_keys.argsort()
        global_leaf_keys = np.ascontiguousarray(global_leaf_keys[ind])
        global_num_part_leaves = np.ascontiguousarray(global_num_part_leaves[ind])

        # rebuild tree using global leaves
        global_tree = QuadTree(self.global_num_real_particles, self.sorted_keys,
                self.corner, self.box_length,
                global_leaf_keys, global_num_part_leaves,
                total_num_process=self.size, factor=self.factor, order=self.order)
        global_tree.build_tree()

        return global_tree

    def calculate_global_work(self, global_tree):
        """Calculate global work by calculating local work in each leaf. Then sum
        work across all process. Currently the work is just the the number of
        particles in each leaf.
        """
        # map each leaf to an array index  
        num_leaves = global_tree.assign_leaves_to_array()
        work = np.zeros(num_leaves, dtype=np.int32)

        # work is just the number of local particles in each leaf
        global_tree.calculate_work(self.keys, work)
        self.global_work = np.empty(num_leaves, dtype=np.int32)

        # collect work from all process
        self.comm.Allreduce(sendbuf=work, recvbuf=self.global_work, op=MPI.SUM)

    def find_split_in_work(self):
        """Partition the global leaves amongst the process such that each process
        has roughly equal work load.
        """
        cumsum_bins = np.cumsum(self.global_work)
        part_per_proc = np.float64(cumsum_bins[-1])/self.size
        leaves_start  = np.zeros(self.size, dtype=np.int32)
        leaves_end    = np.zeros(self.size, dtype=np.int32)
        self.leaf_proc = np.empty(self.global_work.size, dtype=np.int32)

        j = 0
        for i in range(1, self.size):
            end = np.argmax(cumsum_bins > i*part_per_proc)
            leaves_start[i] = end
            leaves_end[i-1] = end - 1
            self.leaf_proc[j:end+1] = i-1
            j = end

        leaves_end[-1] = self.global_work.size-1
        self.leaf_proc[j:] = self.size-1

    def create_boundary_particles(self, pc, rank):
        self.global_tree.create_boundary_particles(pc, rank, self.leaf_proc)

    def update_particle_process(self, pc, rank):
        self.global_tree.update_particle_process(pc, rank, self.leaf_proc)

    def update_particle_domain_info(self, pc, rank):
        self.global_tree.update_particle_domain_info(pc, rank, self.leaf_proc)

    def flag_migrate_particles(self, pc, rank):
        self.global_tree.flag_migrate_particles(pc, rank, self.leaf_proc)
