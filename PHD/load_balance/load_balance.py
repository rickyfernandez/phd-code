import numpy as np

from mpi4py import MPI
from .tree import QuadTree
from hilbert.hilbert import hilbert_key_2d


class LoadBalance(object):

    def __init__(self, particles, order=21):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        self.order = order
        self.number_particles = particles.shape[1]

        xmin = np.empty(2, dtype=np.float64)
        xmax = np.empty(2, dtype=np.float64)
        global_xmin = np.empty(2, dtype=np.float64)
        global_xmax = np.empty(2, dtype=np.float64)

        for i, axis in enumerate(particles):
            xmin[i] = np.min(axis)
            xmax[i] = np.max(axis)

#        if rank == 3:
#            print "min: %s" % xmin
#            print "max: %s" % xmax

        comm.Allreduce(sendbuf=xmin, recvbuf=global_xmin, op=MPI.MIN)
        comm.Allreduce(sendbuf=xmax, recvbuf=global_xmax, op=MPI.MAX)

#        if rank == 3:
#            print "global min: %s" % global_xmin
#            print "global max: %s" % global_xmax

        length = np.max(global_xmax - global_xmin)
        length *= 1.001

        corner = 0.5*(global_xmin + global_xmax) - 0.5*length

        part = (particles - corner[:,np.newaxis])*2**order/length
#        if rank == 0:
#            print "proc: %d normalized particles %s" % (rank, part)

        self.keys = np.array([hilbert_key_2d(p[0], p[1], order)
            for p in (part.T).astype(dtype=np.int64)])

#        if rank == 0:
#            print "keys: %s" % self.keys

        self.sorted_keys = np.empty_like(self.keys)
        np.copyto(self.sorted_keys, self.keys)
        self.sorted_keys.sort()


    def decomposition(self):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        self.build_global_tree()
        self.find_work_done()
        self.find_split_in_work()

        # count the number of particles that need to be exported
        num_export = self.global_tree.count_particles_export(self.keys, self.leaf_proc, rank)

        #print "proc: %s num export: %s" % (rank, num_export)

        export_ids = np.empty(num_export, dtype=np.int32)
        export_proc = np.empty(num_export, dtype=np.int32)

        #collect particles to be exported with theri process location
        self.global_tree.collect_particles_export(self.keys, export_ids, export_proc, self.leaf_proc, rank)
        #print "proc: %s export ids: %s export proc: %s" % (rank, export_ids, export_proc)

        print "proc: %s keys to export: %s orig keys: %s proc id: %s" %\
                (rank, self.keys[export_ids], self.keys, export_proc)

        return {"number" : num_export, "ids": export_ids, "procs": export_proc}


    def get_bounding_box(self):
        pass

    def build_global_tree(self):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # collect number of particles from all process
        sendbuf = np.array([self.number_particles], dtype=np.int32)
        proc_num_particles = np.empty(size, dtype=np.int32)

        comm.Allgather(sendbuf=sendbuf, recvbuf=proc_num_particles)

        global_tot_num_particles = np.sum(proc_num_particles)

#        print "proc %d: local particles %d total particles %s" %\
#                (rank, self.number_particles, proc_num_particles)

        # construct local tree
        local_tree = QuadTree(global_tot_num_particles, self.sorted_keys,
                total_num_process=size, order=self.order)
        local_tree.build_tree()

#        print "proc %d: number leaves %d  number of nodes %s" %\
#                (rank, local_tree.count_leaves(), local_tree.count_nodes())

        # collect leaves to send to all process
        leaf_keys, num_part_leaf = local_tree.collect_leaves_for_export()

#        print "proc %d: leaf keys %s num_part_leaf %s" %\
#                (rank, leaf_keys, num_part_leaf)

        counts = np.empty(size, dtype=np.int32)
        sendbuf = np.array([leaf_keys.size], dtype=np.int32)
        comm.Allgather(sendbuf=sendbuf, recvbuf=counts)

#        print "proc %d: num leaf keys %s counts %s" %\
#                (rank, leaf_keys.size, counts)
#
        global_num_leaves = np.sum(counts)
        offsets = np.zeros(size)
        offsets[1:] = np.cumsum(counts)[:-1]

        global_leaf_keys = np.empty(global_num_leaves, dtype=np.int64)
        global_num_part_leaves = np.empty(global_num_leaves, dtype=np.int32)

        comm.Allgatherv(leaf_keys, [global_leaf_keys, counts, offsets, MPI.INT64_T])
        comm.Allgatherv(num_part_leaf, [global_num_part_leaves, counts, offsets, MPI.INT])

#        print "proc %d: global leaf keys %d counts %s" %\
#                (rank, global_leaf_keys.size, counts)

        # sort the hilbert segments
        ind = global_leaf_keys.argsort()
        global_leaf_keys = np.ascontiguousarray(global_leaf_keys[ind])
        global_num_part_leaves = np.ascontiguousarray(global_num_part_leaves[ind])

        # rebuild tree using global leaves
        self.global_tree = QuadTree(global_tot_num_particles, self.sorted_keys,
                global_leaf_keys, global_num_part_leaves,
                total_num_process=size, order=self.order)
        self.global_tree.build_tree()

        # collect leaves to send to all process
        leaf_keys, num_part_leaf = self.global_tree.collect_leaves_for_export()
        if rank == 0:
            print "proc %d: key leaves %s" % (rank, leaf_keys)
#        print "proc %d: number leaves %d  number of nodes %s" %\
#                (rank, self.global_tree.count_leaves(), self.global_tree.count_nodes())


    def find_work_done(self):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        num_leaves = self.global_tree.assign_leaves_to_array()
        work = np.zeros(num_leaves, dtype=np.int32)

#        print "proc %d: num leaves %s" % (rank, num_leaves)

        self.global_tree.calculate_work(self.keys, work)
        self.global_work = np.empty(num_leaves, dtype=np.int32)

#        print "proc %d: local work %s" % (rank, work)
#
#        # collect work from all process
        comm.Allreduce(sendbuf=work, recvbuf=self.global_work, op=MPI.SUM)

        if rank == 0:
            print "proc %d: global work %d  array %s" %\
                    (rank, np.sum(self.global_work), self.global_work)

#
    def find_split_in_work(self):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        cumsum_bins = np.cumsum(self.global_work)
        part_per_proc = np.float64(cumsum_bins[-1])/size
        leaves_start = np.zeros(size, dtype=np.int32)
        leaves_end = np.zeros(size, dtype=np.int32)
        self.leaf_proc = np.empty(self.global_work.size, dtype=np.int32)

#        if rank == 0:
#            print "part per proc: %s" % part_per_proc

        j = 0
        for i in range(1, size):
            end = np.argmax(cumsum_bins > i*part_per_proc)
            leaves_start[i] = end
            leaves_end[i-1] = end - 1
            self.leaf_proc[j:end+1] = i-1
            j = end

        leaves_end[-1] = self.global_work.size-1
        self.leaf_proc[j:] = size-1

        if rank == 0:
            print "rank: %d leaf procs %s" % (rank, self.leaf_proc)

#        if rank == 0:
#            print "work: %s procs: %s leaves start: %s leaves end: %s" %\
#                    (self.global_work, self.leaf_proc, leaves_start, leaves_end)
