import numpy as np

cimport numpy as np
cimport libc.stdlib as stdlib

from cython.operator cimport dereference as deref, preincrement as inc

cdef find_boundary_particles(neighbor_graph, cumsum_neighbors, ghost_indices, total_ghost_indices):
    """Find border particles, two layers, and return their indicies.
    """

    cdef set[np.uint32_t] interior_id
    cdef set[np.uint32_t].iterator it

    # create initial layer of boundary particles
    cdef i, j, k
    for i in ghost_indices:

        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]

        for j in xrange(start, end):
            k = neighbor_graph[j]
            # is the particle an interior particle 
            if particle_id[k] == 0:
                # no duplicates
                if interior_id.find(j) == interior_id.end():
                    interior_id.insert(j)

    cdef set[np.uint32_t] final_boundary_id

    # add first layer of particles first
    it = interior_id.begin()
    while it != interior_id.end():

        i = deref(it)
        final_boundary_id.insert(i)

    # now add the second boundary
    it = interior_id.begin()
    while it != interior_id.end():

        i = deref(it)
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]

        for j in xrange(start, end):
            k = neighbor_graph[j]
            if particle_id[k] == 0:
                if final_interior_id.find(j) == final_interior_id.end():
                    final_interior_id.insert(j)

    # create numpy array
    return np.array(list(border))

cdef indices_for_export(neighbor_graph, neighbors_graph_size, ghost_indices, total_ghost_indices):
    """Find border particles, two layers, and return their indicies.
    """
    cumsum_neighbors = neighbors_graph_size.cumsum()

    # load all ghost particles to the set
    cdef set[np.uint32_t] interior_id
    cdef set[np.uint32_t].iterator it

    cdef i, j, k
    for i in ghost_indices:

        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]

        for j in xrange(start, end):
            k = neighbor_graph[j]
            if particle_id[k] == 0: # particle is real
                if interior_id.find(j) == interior_id.end():
                    interior_id.insert(j)

    cdef set[np.uint32_t] final_boundary_id

    # add old particles first
    it = interior_id.begin()
    while it != interior_id.end():

        i = deref(it)
        final_boundary_id.insert(i)

    # now add the second boundary
    it = interior_id.begin()
    while it != interior_id.end():

        i = deref(it)
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]

        for j in xrange(start, end):
            k = neighbor_graph[j]
            if particle_id[k] == 0: # particle is real
                if final_interior_id.find(j) == final_interior_id.end(): # no duplicates
                    final_interior_id.insert(j)

    # create numpy array
    return np.array(list(border))
