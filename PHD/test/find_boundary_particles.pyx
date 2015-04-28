import numpy as np

cimport numpy as np
cimport libc.stdlib as stdlib

from cython.operator cimport dereference as deref, preincrement as inc

cdef find_boundary_particles(neighbor_graph, neighbors_graph_size, ghost_indices, total_ghost_indices):
    """Find border particles, two layers, and return their indicies.
    """
    cumsum_neighbors = neighbors_graph_size.cumsum()

    # load all ghost particles to the set
    cdef set[np.uint32_t] boundary_id
    cdef set[np.uint32_t] interior_id
    cdef set[np.uint32_t].iterator it

    cdef i, j
    for i in ghost_indices:

        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]

        for j in xrange(start, end):

            # is the particle an interior particle 
            if boundary_id.find(j) == boundary_id.end():
                # no duplicates
                if interior_id.find(j) == interior_id.end():
                    interior_id.insert(j)

    cdef set[np.uint32_t] final_boundary_id

    it = interior_id.begin()
    while it != interior_id.end():

        i = deref(it)
        if final_interior_id.find(i) == final_interior_id.end():
            final_boundary_id.insert(i)

        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]

        for j in xrange(start, end):
            # no duplicates
            if final_interior_id.find(j) == final_interior_id.end():
                final_interior_id.insert(j)

    # create numpy array

    return np.array(list(border))
