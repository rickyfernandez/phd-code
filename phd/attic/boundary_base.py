import numpy as np

class BoundaryBase(object):
    """
    boundary condition base class, every boundary class must inherit
    this class
    """
    def __init__(self, left, right, bottom, top):

        self.dim = None
        self.boundaries = None


    def update_boundaries(self, particles, particles_index, neighbor_graph):
        """
        every boundary class must have an update method which generates
        ghost particles from real particles
        """
        pass


    def find_boundary_particles(self, neighbor_graph, neighbors_graph_size, ghost_indices, total_ghost_indices):
        """
        find border particles, two layers, and return their indicies. This works in 2d and 3d.
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


    def primitive_to_ghost(self, particles, primitive, particles_index):
        """
        copy primitive values from real particles to associated ghost particles. This works in 2d and 3d.
        """
        ghost_map = particles_index["ghost_map"]
        return np.hstack((primitive, primitive[:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))


    def gradient_to_ghost(self, particles, grad, particles_index):
        """
        copy gradient values from real particles to associated ghost particles. This works in 2d and 3d.
        """
        ghost_map = particles_index["ghost_map"]

        # new gradient with values for real and ghost particles
        new_grad = {}
        for key in grad.keys():
            new_grad[key] = np.hstack((grad[key], grad[key][:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))

        return new_grad
