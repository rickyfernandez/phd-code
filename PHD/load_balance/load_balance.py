import numpy as np

from mpi4py import MPI
from .tree import QuadTree
from hilbert.hilbert import hilbert_key_2d

# maybe turn this into a function?
class LoadBalance(object):

    def __init__(self, particles, corner=None, box_length=1.0, comm=None, factor=0.1, order=21):
        """Constructor for load balance

        Parameters
        ----------
        particles - particle container
        corner - left corner of the box simulation
        box_length - size of the box simulation
        comm - MPI.COMM_WORLD for communication
        order - the number of bits per dimension for constructing
                hilbert keys
        """
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.factor = factor

        self.order = order
        self.number_real_particles = particles.num_real_particles
        self.particles = particles
        self.box_length = box_length

        if corner is None:
            self.corner = np.zeros(2, dtype=np.float64)
        else:
            self.corner = corner

        self.keys = None
        self.sorted_keys = None
        #self.global_num_particles = None
        self.global_num_real_particles = None

    def calculate_hilbert_keys(self):
        """map particle positions to hilbert space
        """
        # normalize coordinates to hilbert space
        fac = (1 << self.order)/ self.box_length
        x = self.particles['position-x']
        y = self.particles['position-y']

        # create hilbert key for each particle
        self.keys = np.array([hilbert_key_2d(
            np.int64((x[i]-self.corner[0])*fac),
            np.int64((y[i]-self.corner[1])*fac), self.order) for i in range(self.number_real_particles)],
            dtype = np.int64)

        self.sorted_keys = np.sort(self.keys)

    def create_ghost_particles(self):
        """Create initial ghost particles that hug the boundary
        """
        ghost_particles = self.global_tree.create_boundary_particles(self.rank. self.leaf_proc)

    def decomposition(self):
        """Perform a domain decomposition
        """
        self.calculate_hilbert_keys()
        self.build_global_tree()
        self.calculate_global_work()
        self.find_split_in_work()

        # count the number of particles that need to be exported
        num_export = self.global_tree.count_particles_export(self.keys, self.leaf_proc, self.rank)

        self.export_ids = np.empty(num_export, dtype=np.int32)
        self.export_proc = np.empty(num_export, dtype=np.int32)

        # collect particles to be exported with their process location
        self.global_tree.collect_particles_export(self.keys, self.export_ids, self.export_proc, self.leaf_proc, self.rank)

    def build_global_tree(self):
        """Build a global tree on all process. This algorithm follows what springel
        outlines in (2005). We first create a local tree from local particles by
        subdividing the hilbert keys. Then all the local leaves are collected to all
        process and a second tree is created using the leaves (i.e. the hilbert cuts).
        """
        # collect number of particles from all process
        sendbuf = np.array([self.number_real_particles], dtype=np.int32)
        proc_num_real_particles = np.empty(self.size, dtype=np.int32)

        self.comm.Allgather(sendbuf=sendbuf, recvbuf=proc_num_real_particles)
        self.global_num_real_particles = np.sum(proc_num_real_particles)

        # construct local tree
        local_tree = QuadTree(self.global_num_real_particles, self.sorted_keys,
                total_num_process=self.size, factor=self.factor, order=self.order)
        local_tree.build_tree()

        # collect leaf start keys and number of particles in leaf and send to all process
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
        self.global_tree = QuadTree(self.global_num_real_particles, self.sorted_keys,
                global_leaf_keys, global_num_part_leaves,
                total_num_process=self.size, factor=self.factor, order=self.order)
        self.global_tree.build_tree()

    def calculate_global_work(self):
        """Calculate the global work by calculating the local work in each
        leaf. Then sum the work for each leaf across all process. Currently
        the work is just the the number of particles in each leaf.
        """
        # number global leaves in 1d array
        num_leaves = self.global_tree.assign_leaves_to_array()
        work = np.zeros(num_leaves, dtype=np.int32)

        # work is just the number of local particles in each leaf
        self.global_tree.calculate_work(self.keys, work)
        self.global_work = np.empty(num_leaves, dtype=np.int32)

        # collect work from all process
        self.comm.Allreduce(sendbuf=work, recvbuf=self.global_work, op=MPI.SUM)

    def find_split_in_work(self):
        """Partition the global leaves amongst the process such that each process
        has roughly an equal work load.
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

    def exchange_particles(self):
        """Perform the exchange of particles. Can only be performed after
        the domain decomposition has been performed.
        """
        # arrange particles in process order
        ind = self.export_proc.argsort()
        self.export_proc = self.export_proc[ind]
        self.export_ids  = self.export_ids[ind]

        # count the number of particles to send to each process
        send_particles = np.bincount(self.export_proc,
                minlength=self.size).astype(np.int32)

        # extract data to send and remove the particles
        send_data = {}
        for prop in self.particles.properties.keys():
            send_data[prop] = self.particles[prop][self.export_ids]

        # remove exported and ghost particles
        self.particles.discard_ghost_and_export_particles(self.export_ids)

        # how many particles are being sent from each process
        recv_particles = np.empty(self.size, dtype=np.int32)
        self.comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)

        # resize arrays to give room for incoming particles
        current_real_size = self.particles.num_real_particles
        new_real_size = current_real_size + np.sum(recv_particles)
        self.particles.resize(new_real_size)

        # displacements for the send and reveive buffers
        offset_se = np.zeros(self.size, dtype=np.int32)
        offset_re = np.zeros(self.size, dtype=np.int32)
        for i in range(1,self.size):
            offset_se[i] = send_particles[i-1] + offset_se[i-1]
            offset_re[i] = recv_particles[i-1] + offset_re[i-1]

        ptask = 0
        while self.size > (1<<ptask):
            ptask += 1

        for ngrp in xrange(1,1 << ptask):
            sendTask = self.rank
            recvTask = self.rank ^ ngrp
            if recvTask < self.size:
                if send_particles[recvTask] > 0 or recv_particles[recvTask] > 0:
                    for prop in self.particles.properties.keys():

                        sendbuf=[send_data[prop],   (send_particles[recvTask], offset_se[recvTask])]
                        recvbuf=[self.particles[prop][current_real_size:], (recv_particles[recvTask],
                            offset_re[recvTask])]

                        self.comm.Sendrecv(sendbuf=sendbuf, dest=recvTask, recvbuf=recvbuf, source=recvTask)

        # update the new number of real particles in the container
        self.particles.num_real_particles = new_real_size
