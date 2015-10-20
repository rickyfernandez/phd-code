import numpy as np

from utils.particle_tags import ParticleTAGS
from utils.carray import IntArray, LongArray
from mesh.voronoi_mesh import VoronoiMesh2D
from utils.exchange_particles import exchange_particles

class SingleCoreBoundary(object):

    def update_ghost_particles(self, particles, mesh, domain):

        # relabel all particles
        self.flag_exterior_particles(particles, domain.xmin, domain.xmax)

        # create new exterior boundary ghost particles
        cumsum_neighbors = mesh["number of neighbors"].cumsum()
        ghost_indices = np.where(particles["tag"] == ParticleTAGS.OldGhost)[0]
        num_exterior_ghost = create_reflect_ghost(particles, domain, ghost_indices, ghost_indices,
                mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors)

        # remove old ghost particles
        particles.remove_tagged_particles(ParticleTAGS.OldGhost)

        # place real particles in begginning of the array
        particles.align_particles()
        particles['type'][:]  = ParticleTAGS.Undefined

    def flag_exterior_particles(self, pc, lo, hi):

        x = pc['position-x']
        y = pc['position-y']

        # find all particles in the interior domain
        indices = (((lo <= x) & (x <= hi)) & ((lo <= y) & (y <= hi)))
        pc['tag'][indices]  = ParticleTAGS.Real
        pc['tag'][~indices] = ParticleTAGS.OldGhost


#class MultiCoreBoundary(object):
#
#    def create_ghost_particles(self, particles, leaf_proc, global_tree,
#            domain, comm, iteration=5):
#        """Create initial ghost particles that hug the boundary
#        """
#        rank = comm.Get_rank()
#        size = comm.Get_size()
#
#        # remove current (if any) ghost particles
#        particles.remove_tagged_particles(ParticleTAGS.Ghost)
#        current_size = particles.get_number_of_particles()
#
#        # create initial ghost particles, particles is now larger
#        global_tree.create_boundary_particles(particles, rank, leaf_proc)
#
#        # reorder ghost in processors order: exterior have a process id of -1 so
#        # their put before interior ghost particles
#        ghost_proc = np.array(particles["process"][current_size:])
#        ind = np.argsort(ghost_proc)
#        ghost_proc = ghost_proc[ind]
#
#        for field in particles.properties.keys():
#            array = particles[field][current_size:]
#            array[:] = array[ind]
#
#        mesh = VoronoiMesh2D()
#        indices = LongArray()
#
#        # create ghost interior and exterior particles by iteration, using
#        # the mesh to extract the needed neighbors
#        for i in range(iteration):
#
#            # build the mesh
#            p = np.array([particles['position-x'], particles['position-y']])
#            mesh.tessellate(p)
#            cumsum_neighbors = mesh["number of neighbors"].cumsum()
#
#            # create indices for ghost particles
#            ghost_indices = np.arange(current_size, particles.get_number_of_particles())
#
#            # select exterior ghost particles
#            exterior_ghost = ghost_proc == -1
#            num_exterior_ghost = exterior_ghost.sum()
#            exterior_ghost_indices = ghost_indices[exterior_ghost]
#
#            # create exterior boundary ghost particles
#            num_exterior_ghost = create_reflect_ghost(particles, domain, exterior_ghost_indices, ghost_indices,
#                    mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors)
#
#            # do the interior particles
#            interior_ghost_indices = ghost_indices[~exterior_ghost]
#            interior_ghost_proc = ghost_proc[~exterior_ghost]
#
#            # bin processors
#            interior_ghost_proc_bin = np.bincount(interior_ghost_proc, minlength=size)
#            send_particles = np.zeros(size, dtype=np.int32)
#
#            # collect the indices of particles to be export to each process
#            indices.reset()
#            cumsum_proc = interior_ghost_proc_bin.cumsum()
#            for proc in range(size):
#                if interior_ghost_proc_bin[proc] != 0:
#
#                    start = cumsum_proc[proc] - interior_ghost_proc_bin[proc]
#                    end   = cumsum_proc[proc]
#
#                    send_particles[proc] = find_boundary_particles(indices, interior_ghost_indices[start:end], ghost_indices,
#                            mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, False)
#
#            # extract data to send and remove the particles
#            send_data = {}
#            for prop in particles.properties.keys():
#                send_data[prop] = np.ascontiguousarray(particles[prop][indices.get_npy_array()])
#            send_data["tag"][:] = ParticleTAGS.Ghost
#
#            # how many particles are being sent from each process
#            recv_particles = np.empty(size, dtype=np.int32)
#            comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)
#
#            # resize arrays to give room for incoming particles
#            particles.resize(current_size + num_exterior_ghost + np.sum(recv_particles))
#
#            #print "rank: %d iteration: %d num: %d" % (rank, i, np.sum(recv_particles))
#
#            exchange_particles(particles, send_data, send_particles, recv_particles,
#                    current_size + num_exterior_ghost, comm)
#
#            ghost_proc = np.array(particles["process"][current_size:])
#
#    def update_ghost_particles(self, particles, mesh, leaf_proc, global_tree,
#            domain, comm):
#
#        rank = comm.Get_rank()
#        size = comm.Get_size()
#
#
#        # relabel all particles
#        global_tree.update_boundary_particles(particles, rank, leaf_proc)
#        ghost_indices = np.where(particles["tag"] == ParticleTAGS.OldGhost)[0]
#
#        # create new exterior boundary ghost particles
#        cumsum_neighbors = mesh["number of neighbors"].cumsum()
#        exterior_ghost_indices = np.where(particles["type"] == ParticleTAGS.Exterior)[0]
#        num_exterior_ghost = create_reflect_ghost(particles, domain, exterior_ghost_indices, ghost_indices,
#                mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors)
#
#        # import/export interior ghost particles
#        interior_ghost_indices = np.where(
#                (particles["type"] == ParticleTAGS.Interior) | (particles["type"] == ParticleTAGS.ExportInterior))[0]
#        interior_ghost_proc = particles["process"][interior_ghost_indices]
#
#        # arrange particles in process order
#        ind = interior_ghost_proc.argsort()
#        interior_ghost_proc = interior_ghost_proc[ind]
#        interior_ghost_indices = interior_ghost_indices[ind]
#
#        # bin processors
#        interior_ghost_proc_bin = np.bincount(interior_ghost_proc, minlength=size)
#        send_particles = np.zeros(size, dtype=np.int32)
#
#        # collect the indices of particles to be export to each process
#        indices = LongArray()
#        cumsum_proc = interior_ghost_proc_bin.cumsum()
#        for proc in range(size):
#            if interior_ghost_proc_bin[proc] != 0:
#
#                start = cumsum_proc[proc] - interior_ghost_proc_bin[proc]
#                end   = cumsum_proc[proc]
#
#                send_particles[proc] = find_boundary_particles(indices, interior_ghost_indices[start:end], ghost_indices,
#                        mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, False)
#
#        # extract data to send and remove the particles
#        send_data = {}
#        for prop in particles.properties.keys():
#            send_data[prop] = np.ascontiguousarray(particles[prop][indices.get_npy_array()])
#
#        # these particles are now ghost particles for the process their going to
#        send_data["tag"][:] = ParticleTAGS.Ghost
#
#        # how many particles are being sent from each process
#        recv_particles = np.empty(size, dtype=np.int32)
#        comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)
#
#        # resize arrays to give room for incoming particles
#        current_size = particles.get_number_of_particles()
#        particles.extend(np.sum(recv_particles))
#
#        exchange_particles(particles, send_data, send_particles, recv_particles,
#                current_size, comm)
#
#        # export real particles that left the domain
#        export_ghost_indices = np.where(particles["tag"] == ParticleTAGS.ExportInterior)[0]
#        export_ghost_proc = particles["process"][export_ghost_indices]
#
#        # arrange particles in process order
#        ind = export_ghost_proc.argsort()
#        export_ghost_proc = export_ghost_proc[ind]
#        export_ghost_indices  = export_ids[ind]
#
#        # count the number of particles to send to each process
#        send_particles = np.bincount(export_ghost_proc,
#                minlength=size).astype(np.int32)
#
#        # extract particles to send 
#        send_data = particles.get_sendbufs(export_ghost_indices)
#        send_data["tag"][:] = ParticleTAGS.Real
#
#        # how many particles are being sent from each process
#        recv_particles = np.empty(size, dtype=np.int32)
#        comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)
#
#        # resize particle array and place incoming particles at the
#        # end of the array
#        current_size = particles.get_number_of_particles()
#        particles.extend(np.sum(recv_particles))
#
#        # exchange load balance particles
#        exchange_particles(particles, send_data, send_particles, recv_particles,
#                current_size, comm)
#
#        # finally remove old ghost particles from previous time step
#        # this also puts real particles in front and ghost in the back
#        particles.remove_tagged_particles(ParticleTAGS.OldGhost)


def create_reflect_ghost(parray, domain, exterior_ghost_indices, ghost_indices,
        neighbors_graph, neighbors_graph_size, cumsum_neighbors):

    indices = LongArray()

    left = LongArray()
    right = LongArray()
    bottom = LongArray()
    top = LongArray()

    xmin = domain.xmin
    xmax = domain.xmax
    ymin = domain.ymin
    ymax = domain.ymax

    x = parray['position-x']
    y = parray['position-y']

    for i in exterior_ghost_indices:

        xi = x[i]; yi = y[i]

        # left boundary condition
        if xi < xmin:
            left.append(i)

        # right boundary condition
        if xi > xmax:
            right.append(i)

        # bottom boundary condition
        if yi < ymin:
            bottom.append(i)

        # top boundary condition
        if yi > ymax:
            top.append(i)

    num_new_ghost = 0
    parray.remove_tagged_particles(ParticleTAGS.Ghost)


    # find corresponding real particles to make ghost
    #find_boundary_particles(indices, left.get_npy_array(), ghost_indices,
    find_boundary_particles(indices, left.get_npy_array(), left.get_npy_array(),
            neighbors_graph, neighbors_graph_size, cumsum_neighbors)
    copy = parray.extract_items(indices.get_npy_array())
    num_new_ghost += copy.get_number_of_items()
    copy['position-x'][:] -= 2*(copy['position-x'] - xmin)
    copy['tag'][:] = ParticleTAGS.Ghost
    copy['process'][:] = -1
    parray.append_container(copy)

    #find_boundary_particles(indices, right.get_npy_array(), ghost_indices,
    find_boundary_particles(indices, right.get_npy_array(), right.get_npy_array(),
            neighbors_graph, neighbors_graph_size, cumsum_neighbors)
    copy = parray.extract_items(indices.get_npy_array())
    num_new_ghost += copy.get_number_of_items()
    copy['position-x'][:] += 2*(xmax - copy['position-x'])
    copy['tag'][:] = ParticleTAGS.Ghost
    copy['process'][:] = -1
    parray.append_container(copy)

    find_boundary_particles(indices, bottom.get_npy_array(), ghost_indices,
            neighbors_graph, neighbors_graph_size, cumsum_neighbors)
    copy = parray.extract_items(indices.get_npy_array())
    num_new_ghost += copy.get_number_of_items()
    copy['position-y'][:] -= 2*(copy['position-y']-ymin)
    copy['tag'][:] = ParticleTAGS.Ghost
    copy['process'][:] = -1
    parray.append_container(copy)

    find_boundary_particles(indices, top.get_npy_array(), ghost_indices,
            neighbors_graph, neighbors_graph_size, cumsum_neighbors)
    copy = parray.extract_items(indices.get_npy_array())
    num_new_ghost += copy.get_number_of_items()
    copy['position-y'][:] += 2*(ymax - copy['position-y'])
    copy['tag'][:] = ParticleTAGS.Ghost
    copy['process'][:] = -1
    parray.append_container(copy)

    return num_new_ghost


def find_boundary_particles(indices, ghost_indices, total_ghost_indices,
        neighbors_graph, neighbors_graph_size, cumsum_neighbors, clean=True):
    """
    Find border particles, two layers, and return their indicies.
    """

    if clean:
        indices.reset()

    # grab all neighbors of ghost particles, this includes border cells
    border = set()
    for i in ghost_indices:
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]
        border.update(neighbors_graph[start:end])

    # grab neighbors again, this includes another layer of border cells 
    border_tmp = set(border)
    for i in border_tmp:
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]
        border.update(neighbors_graph[start:end])

    border_tmp = set(border)
    for i in border_tmp:
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]
        border.update(neighbors_graph[start:end])

    # remove ghost particles leaving border cells that will create new
    # ghost particles
    border = border.difference(total_ghost_indices)

    for i in border:
        indices.append(i)

    return len(border)
