import numpy as np

class boundary_base(object):

    def update(self, particles, particles_index, neighbor_graph):
        pass

    def find_boundary_particles(self, neighbor_graph, neighbors_graph_size, ghost_indices, total_ghost_indices):
        """
        find border particles, two layers, and return their indicies
        """

        cumsum_neighbors = neighbors_graph_size.cumsum()

        # grab all neighbors of ghost particles, this includes border cells
        border = set()
        for i in ghost_indices:
            start = cumsum_neighbors[i] - neighbors_graph_size[i]
            end   = cumsum_neighbors[i]
            border.update(neighbor_graph[start:end])

        # grab neighbors of border cells 
        border_tmp = set(border)
        for i in border_tmp:
            start = cumsum_neighbors[i] - neighbors_graph_size[i]
            end   = cumsum_neighbors[i]
            border.update(neighbor_graph[start:end])

        # remove ghost particles leaving border cells and neighbors, two layers
        border = border.difference(total_ghost_indices)

        return np.array(list(border))

    def primitive_to_ghost(self, particles, primitive, particles_index):
        """
        copy primitive values from real particles to associated ghost particles
        """
        ghost_map = particles_index["ghost_map"]
        return np.hstack((primitive, primitive[:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))

