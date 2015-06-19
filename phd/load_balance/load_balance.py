import itertools
import numpy as np
from mpi4py import MPI

from .tree import QuadTree
from particles.particle_tags import ParticleTAGS
from hilbert.hilbert import hilbert_key_2d
from mesh.voronoi_mesh_2d import VoronoiMesh2D
from utils.exchange_particles import exchange_particles


def find_boundary_particles(neighbor_graph, neighbors_graph_size, ghost_indices, total_ghost_indices):
    """Find border particles, two layers, and return their indicies.
    """
    cumsum_neighbors = neighbors_graph_size.cumsum()

    # grab all neighbors of ghost particles, this includes border cells
    border = set()
    for i in ghost_indices:
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]
        border.update(neighbor_graph[start:end])

    # grab neighbors again, this includes another layer of border cells 
    border_tmp = set(border)
    for i in border_tmp:
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]
        border.update(neighbor_graph[start:end])

    # remove ghost particles leaving border cells that will create new ghost particles
    border = border.difference(total_ghost_indices)

    return np.array(list(border))

class LoadBalance(object):

    def __init__(self, parray, domain, comm=None, factor=0.1, order=21):
        """Constructor for load balance

        Parameters
        ----------
        parray : ParticleArray
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
        self.number_real_particles = parray.num_real_particles
        self.parray = parray

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

    def calculate_hilbert_keys(self):
        """map particle positions to hilbert space"""

        x = self.parray.get('position-x')
        y = self.parray.get('position-y')

        # normalize coordinates to hilbert space
        fac = (1 << self.order)/ self.box_length

        # create hilbert key for each particle
        self.keys = np.array([hilbert_key_2d(
            np.int64((x[i]-self.corner[0])*fac),
            np.int64((y[i]-self.corner[1])*fac), self.order) for i in range(self.number_real_particles)],
            dtype = np.int64)

        self.sorted_keys = np.sort(self.keys)

    # turn this into a boundary class
#    def create_ghost_particles(self, global_tree):
#        """Create initial ghost particles that hug the boundary
#        """
#        ghost_particles = global_tree.create_boundary_particles(self.rank, self.leaf_proc)
#        ghost_particles = np.transpose(np.array(ghost_particles))
#
#        # reorder in processors order: exterior are put before interior ghost particles
#        proc_id = ghost_particles[2,:].astype(np.int32)
#        ind = np.argsort(proc_id)
#        ghost_particles = ghost_particles[:2,ind]
#        proc_id = proc_id[ind]
#
#        # create new position array for temporary mesh
#        self.particles.discard_ghost_particles()
#        current_size = self.particles.num_real_particles
#        new_size = current_size + ghost_particles.shape[1]
#        self.particles.resize(new_size)
#
#        # add this to the particle container instead
#        self.particles['position-x'][current_size:] = ghost_particles[0,:]/2.0**self.order
#        self.particles['position-y'][current_size:] = ghost_particles[1,:]/2.0**self.order
#
#        boundaries = [[self.corner[0], self.corner[0]+self.box_length],
#                [self.corner[1], self.corner[1]+self.box_length]]
#
#        mesh = VoronoiMesh2D()
#        for i in range(5):
#
#            # build the mesh
#            p = np.array([self.particles['position-x'], self.particles['position-y']])
#            #graphs = mesh.tessellate(p)
#            mesh.tessellate(p)
#
#            ghost_indices = np.arange(current_size, new_size)
#
#            # select exterior ghost particles
#            exterior_ghost = proc_id == -1
#            num_exterior_ghost = np.sum(exterior_ghost)
#            exterior_ghost_indices = ghost_indices[exterior_ghost]
#
#            new_ghost = np.empty((2,0), dtype=np.float64)
#            old_ghost = p[:,current_size:current_size+num_exterior_ghost]
#            num_exterior_ghost = 0 # count new exterior
#
#            for k, bound in enumerate(boundaries):
#
#                do_min = True
#                for qm in bound:
#
#                    if do_min == True:
#                        # lower boundary 
#                        i = np.where(old_ghost[k,:] < qm)[0]
#                        do_min = False
#                    else:
#                        # upper boundary
#                        i = np.where(qm < old_ghost[k,:])[0]
#
#                    # find bordering real particles
#                    border = find_boundary_particles(mesh['neighbors'], mesh['number of neighbors'],
#                            exterior_ghost_indices[i], ghost_indices)
#
#                    if border.size != 0:
#
#                        # allocate space for new ghost particles
#                        tmp = np.empty((2, len(border)), dtype=np.float64)
#
#                        # reflect particles across boundary
#                        tmp[:,:] = p[:,border]
#                        tmp[k,:] = 2*qm - p[k,border]
#
#                        # add the new ghost particles
#                        new_ghost = np.concatenate((new_ghost, tmp), axis=1)
#                        num_exterior_ghost += border.size
#
#            new_size = current_size + num_exterior_ghost
#            self.particles.resize(new_size)
#
#            # add in the new exterior ghost particles
#            self.particles['position-x'][current_size:current_size+num_exterior_ghost] = new_ghost[0,:]
#            self.particles['position-y'][current_size:current_size+num_exterior_ghost] = new_ghost[1,:]
#
#
#            # do the interior particles now
#            interior_ghost_indices = ghost_indices[~exterior_ghost]
#            interior_proc_id = proc_id[~exterior_ghost]
#
#            # delete this
#            init_border = find_boundary_particles(mesh['neighbors'], mesh['number of neighbors'],
#                    interior_ghost_indices, ghost_indices)
#
#            # bin processors
#            interior_proc_bin = np.bincount(interior_proc_id, minlength=self.size)
#            send_particles = np.zeros(self.size, dtype=np.int32)
#
#            # collect the indices for particles to export for each process
#            ghost_list = []
#            cumsum_proc = interior_proc_bin.cumsum()
#            for proc in xrange(self.size):
#                if interior_proc_bin[proc] != 0:
#
#                    start = cumsum_proc[proc] - interior_proc_bin[proc]
#                    end   = cumsum_proc[proc]
#
#                    border = find_boundary_particles(mesh['neighbors'], mesh['number of neighbors'],
#                            interior_ghost_indices[start:end], ghost_indices)
#
#                    ghost_list.append(border)
#                    send_particles[proc] = border.size
#
#            # flatten out the indices
#            new_ghost = np.array(list(itertools.chain.from_iterable(ghost_list)), dtype=np.int32)
#
#            # extract data to send and remove the particles
#            send_data = {}
#            for prop in self.particles.properties.keys():
#                send_data[prop] = np.ascontiguousarray(self.particles[prop][new_ghost])
#
#            # discard current ghost particles: this does not release memory used
#            self.particles.discard_ghost_particles()
#
#            # how many particles are being sent from each process
#            recv_particles = np.empty(self.size, dtype=np.int32)
#            self.comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)
#
#            # resize arrays to give room for incoming particles
#            new_size = current_size + num_exterior_ghost + np.sum(recv_particles)
#            self.particles.resize(new_size)
#
#            #print "rank: %d iteration: %d num: %d" % (self.rank, i, np.sum(recv_particles))
#
#            self.exchange_particles(self.particles, send_data, send_particles, recv_particles,
#                    current_size + num_exterior_ghost)
#
#            # temp hack 
#            proc_id = np.concatenate((-1*np.ones(num_exterior_ghost), np.repeat(np.arange(self.size),
#                recv_particles))).astype(np.int32)
#
#        return init_border
#
#    def update_ghost_particles(self):
#
#        # its after a timestep, particles have moved
#
#        # recalculate hilbert keys, tag them
#
#        # create processor id for particles
#
#        # put ghost particles at the end of the array
#
#        # reorder ghost particles by processor rank
#
#        # generate boundary particles
#
#        # generate a new mesh
#
#        # label border particles

    def decomposition(self):
        """Perform domain decomposition
        """

        # remove current (if any) ghost particles
        self.parray.remove_tagged_particles(ParticleTAGS.Ghost)

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

        # count the number of particles to send to each process
        send_particles = np.bincount(self.export_proc,
                minlength=self.size).astype(np.int32)

        # extract particles to send 
        send_data = {}
        for prop in self.parray.properties.keys():
            send_data[prop] = self.parray[prop][self.export_ids]
        #send_data = extract_particles(self.export_ids)

        # remove exported particles
        self.parray.remove_particles(self.export_ids)

        # how many particles are being sent from each process
        recv_particles = np.empty(self.size, dtype=np.int32)
        self.comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)

        # resize particle array and place incoming particles at the
        # end of the array
        displacement = self.parray.num_real_particles
        num_incoming_particles = np.sum(recv_particles)
        self.parray.extend(num_incoming_particles)

        # exchange load balance particles
        exchange_particles(self.parray, send_data, send_particles, recv_particles,
                displacement, self.comm)

        # align particles and count the number of real particles
        self.parray.align_particles()

        return global_tree

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
