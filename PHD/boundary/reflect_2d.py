import numpy as np
from boundary_base import BoundaryBase

class Reflect2D(BoundaryBase):
    """
    refect boundary class
    """
    def __init__(self, xl, xr, yl, yr):

        self.dim = 2
        self.boundaries = [
                [xl, xr],   # x dim
                [yl, yr]    # y dim
                ]


    def update_boundaries(self, particles, particles_index, neighbor_graph, neighbor_graph_size):
        """
        create ghost particles from real particles using reflective boundary conditins
        """

        ghost_indices = particles_index["ghost"]

        # position of old ghost 
        old_ghost = particles[:,ghost_indices]

        # arrays for position of new ghost particles
        new_ghost = np.empty((self.dim,0), dtype="float64")

        # hold all the indices of real particles used to create ghost particles
        ghost_mapping_indices = np.empty(0, dtype=np.int32)

        # grab all neighbors of ghost particles, this includes border cells and 
        # neighbors of border cells, then remove ghost particles leaving two layers

        for k, bound in enumerate(self.boundaries):

            do_min = True

            for qm in bound:

                if do_min == True:
                    # lower boundary 
                    i = np.where(old_ghost[k,:] < qm)[0]
                    do_min = False
                else:
                    # upper boundary
                    i = np.where(qm < old_ghost[k,:])[0]

                # find bordering real particles
                border = self.find_boundary_particles(neighbor_graph, neighbor_graph_size, ghost_indices[i], ghost_indices)

                # allocate space for new ghost particles
                tmp = np.empty((self.dim, len(border)), dtype="float64")

                # reflect particles across boundary
                tmp[:,:] = particles[:,border]
                tmp[k,:] = 2*qm - particles[k,border]

                # add the new ghost particles
                new_ghost = np.concatenate((new_ghost, tmp), axis=1)

                # add real particles indices correspoinding to ghost particles 
                ghost_mapping_indices = np.append(ghost_mapping_indices, border)

        # create the new list of particles
        real_indices = particles_index["real"]

        # first the real particles
        new_particles = np.copy(particles[:,real_indices])

        # now the new ghost particles
        new_particles = np.concatenate((new_particles, new_ghost), axis=1)

        # update particle information, generate new ghost map
        ghost_map = {}
        for i,j in enumerate(np.arange(real_indices.size, new_particles.shape[1])):
            ghost_map[j] = ghost_mapping_indices[i]

        particles_index["ghost_map"] = ghost_map
        particles_index["ghost"] = np.arange(real_indices.size, new_particles.shape[1])

        return new_particles


    def reverse_velocities(self, particles, primitive, particles_index):
        """
        reflect ghost velocities across the mirror axis
        """

        ghost_indices = particles_index["ghost"]

        # reverse velocities in x direction
        xl = self.boundaries[0][0]
        xr = self.boundaries[0][1]
        x = particles[0,ghost_indices]
        i = np.where((x < xl) | (xr < x))[0]
        primitive[1, ghost_indices[i]] *= -1.0

        # reverse velocities in y direction
        yl = self.boundaries[1][0]
        yr = self.boundaries[1][1]
        y = particles[1,ghost_indices]
        i = np.where((y < yl) | (yr < y))[0]
        primitive[2, ghost_indices[i]] *= -1.0


    def primitive_to_ghost(self, particles, primitive, particles_index):
        """
        copy primitive values to ghost particles from their correponding real particles
        """

        # copy primitive values to ghost
        primitive = super(Reflect2D, self).primitive_to_ghost(particles, primitive, particles_index)

        # ghost particles velocities have to be reversed
        self.reverse_velocities(particles, primitive, particles_index)

        return primitive


    def gradient_to_ghost(self, particles, grad, particles_index):
        """
        copy gradient values to ghost particles from their correponding real particles
        """

        new_grad = super(Reflect2D, self).gradient_to_ghost(particles, grad, particles_index)

        ghost_indices = particles_index["ghost"]

        # reverse velocities in x direction
        #x = particles[0,ghost_indices]
        #i = np.where((x < self.left) | (self.right < x))[0]
        #gradx[1, ghost_indices[i]] *= -1.0
        #grady[1, ghost_indices[i]] *= -1.0

        # reverse velocities in y direction
        #y = particles[1,ghost_indices]
        #i = np.where((y < self.bottom) | (self.top < y))[0]
        #gradx[2, ghost_indices[i]] *= -1.0
        #grady[2, ghost_indices[i]] *= -1.0

        return new_grad
