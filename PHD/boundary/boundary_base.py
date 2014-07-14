import numpy as np

class boundary_base(object):

    def update(self, particles, particles_index, neighbor_graph):
        pass

    def find_boundary_particles(self, neighbor_graph, ghost_indices, total_ghost_indices):

        # find border particles, two layers, and return their indicies 

        # grab all neighbors of ghost particles, this includes border cells
        border = set()
        for i in ghost_indices:
            border.update(neighbor_graph[i])

        # grab neighbors of border cells 
        border_tmp = set(border)
        for i in border_tmp:
            border.update(neighbor_graph[i])

        # remove ghost particles leaving border cells and neighbors
        border = border.difference(total_ghost_indices)
        
        return np.array(list(border))

    def find_boundary_particles2(self, neighbor_graph, num_neighbors, ghost_indices, total_ghost_indices):
        """
        find border particles, two layers, and return their indicies 
        """

        cumsum_neighbors = num_neighbors.cumsum()

        # grab all neighbors of ghost particles, this includes border cells
        border = set()
        for i in ghost_indices:
            start = cumsum_neighbors[i] - num_neighbors[i]
            end   = cumsum_neighbors[i]
            border.update(neighbor_graph[start:end])

        # grab neighbors of border cells 
        border_tmp = set(border)
        for i in border_tmp:
            start = cumsum_neighbors[i] - num_neighbors[i]
            end   = cumsum_neighbors[i]
            border.update(neighbor_graph[start:end])

        # remove ghost particles leaving border cells and neighbors, two layers
        border = border.difference(total_ghost_indices)
        
        return np.array(list(border))
