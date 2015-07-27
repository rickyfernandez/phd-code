import numpy as np

from particles.particle_tags import ParticleTAGS
from utils.carray import IntArray, LongArray
from mesh.voronoi_mesh_2d import VoronoiMesh2D
from utils.exchange_particles import exchange_particles

class Boundary(object):

    def create_ghost_particles(self, parray, leaf_proc, global_tree,
            domain, comm, iteration=5):
        """Create initial ghost particles that hug the boundary
        """
        rank = comm.Get_rank()
        size = comm.Get_size()

        # remove current (if any) ghost particles
        parray.remove_tagged_particles(ParticleTAGS.Ghost)
        current_size = parray.get_number_of_particles()

        # create initial ghost particles, parray is now larger
        global_tree.create_boundary_particles(parray, rank, leaf_proc)

        # reorder ghost in processors order: exterior have a process id of -1 so
        # their put before interior ghost particles
        ghost_proc = np.array(parray["process"][current_size:])
        ind = np.argsort(ghost_proc)
        ghost_proc = ghost_proc[ind]

        for field in parray.properties.keys():
            array = parray[field][current_size:]
            array[:] = array[ind]

        mesh = VoronoiMesh2D()
        indices = LongArray()

        # create ghost interior and exterior particles by iteration, using
        # the mesh to extract the needed neighbors
        for i in range(iteration):

            # build the mesh
            p = np.array([parray['position-x'], parray['position-y']])
            mesh.tessellate(p)
            cumsum_neighbors = mesh["number of neighbors"].cumsum()

            # create indices for ghost particles
            ghost_indices = np.arange(current_size, parray.get_number_of_particles())

            # select exterior ghost particles
            exterior_ghost = ghost_proc == -1
            num_exterior_ghost = exterior_ghost.sum()
            exterior_ghost_indices = ghost_indices[exterior_ghost]

            # create exterior boundary ghost particles
            num_exterior_ghost = create_reflect_ghost(parray, domain, exterior_ghost_indices, ghost_indices,
                    mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors)

            # do the interior particles
            interior_ghost_indices = ghost_indices[~exterior_ghost]
            interior_ghost_proc = ghost_proc[~exterior_ghost]

            # bin processors
            interior_ghost_proc_bin = np.bincount(interior_ghost_proc, minlength=size)
            send_particles = np.zeros(size, dtype=np.int32)

            # collect the indices of particles to be export to each process
            indices.reset()
            cumsum_proc = interior_ghost_proc_bin.cumsum()
            for proc in range(size):
                if interior_ghost_proc_bin[proc] != 0:

                    start = cumsum_proc[proc] - interior_ghost_proc_bin[proc]
                    end   = cumsum_proc[proc]

                    send_particles[proc] = find_boundary_particles(indices, interior_ghost_indices[start:end], ghost_indices,
                            mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, False)

            # extract data to send and remove the particles
            send_data = {}
            for prop in parray.properties.keys():
                send_data[prop] = np.ascontiguousarray(parray[prop][indices.get_npy_array()])
            send_data["tag"][:] = ParticleTAGS.Ghost

            # how many particles are being sent from each process
            recv_particles = np.empty(size, dtype=np.int32)
            comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)

            # resize arrays to give room for incoming particles
            parray.resize(current_size + num_exterior_ghost + np.sum(recv_particles))

            #print "rank: %d iteration: %d num: %d" % (rank, i, np.sum(recv_particles))

            exchange_particles(parray, send_data, send_particles, recv_particles,
                    current_size + num_exterior_ghost, comm)

            ghost_proc = np.array(parray["process"][current_size:])


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
    find_boundary_particles(indices, left.get_npy_array(), ghost_indices,
            neighbors_graph, neighbors_graph_size, cumsum_neighbors)
    copy = parray.extract_particles(indices.get_npy_array())
    num_new_ghost += copy.get_number_of_particles()
    copy['position-x'][:] -= 2*(copy['position-x'] - xmin)
    copy['tag'][:] = ParticleTAGS.Ghost
    copy['process'][:] = -1
    parray.append_parray(copy)

    find_boundary_particles(indices, right.get_npy_array(), ghost_indices,
            neighbors_graph, neighbors_graph_size, cumsum_neighbors)
    copy = parray.extract_particles(indices.get_npy_array())
    num_new_ghost += copy.get_number_of_particles()
    copy['position-x'][:] += 2*(xmax - copy['position-x'])
    copy['tag'][:] = ParticleTAGS.Ghost
    copy['process'][:] = -1
    parray.append_parray(copy)

    find_boundary_particles(indices, bottom.get_npy_array(), ghost_indices,
            neighbors_graph, neighbors_graph_size, cumsum_neighbors)
    copy = parray.extract_particles(indices.get_npy_array())
    num_new_ghost += copy.get_number_of_particles()
    copy['position-y'][:] -= 2*(copy['position-y']-ymin)
    copy['tag'][:] = ParticleTAGS.Ghost
    copy['process'][:] = -1
    parray.append_parray(copy)

    find_boundary_particles(indices, top.get_npy_array(), ghost_indices,
            neighbors_graph, neighbors_graph_size, cumsum_neighbors)
    copy = parray.extract_particles(indices.get_npy_array())
    num_new_ghost += copy.get_number_of_particles()
    copy['position-y'][:] += 2*(ymax - copy['position-y'])
    copy['tag'][:] = ParticleTAGS.Ghost
    copy['process'][:] = -1
    parray.append_parray(copy)

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

    # remove ghost particles leaving border cells that will create new
    # ghost particles
    border = border.difference(total_ghost_indices)

    for i in border:
        indices.append(i)

    return len(border)
