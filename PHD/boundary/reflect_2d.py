import numpy as np
from boundary_base import BoundaryBase

class Reflect(BoundaryBase):
    """
    refect boundary class
    """
    def __init__(self):
        self.dim = 2

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

        for k, bound in enumerate(self.boundaries.itervalues()):

            q = range(self.dim); q = q.remove(k)
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
                tmp = np.empty((self.dim, len(border), dtype="float64")

                # reflect particles across boundary
                tmp[q,:] = particles[q,border]
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
        x = particles[0,ghost_indices]
        i = np.where((x < self.left) | (self.right < x))[0]
        primitive[1, ghost_indices[i]] *= -1.0

        # reverse velocities in y direction
        y = particles[1,ghost_indices]
        i = np.where((y < self.bottom) | (self.top < y))[0]
        primitive[2, ghost_indices[i]] *= -1.0

    def primitive_to_ghost(self, particles, primitive, particles_index):
        """
        copy primitive values to ghost particles from their correponding real particles
        """

        # copy primitive values to ghost
        primitive = super(Reflect, self).primitive_to_ghost(particles, primitive, particles_index)

        # ghost particles velocities have to be reversed
        self.reverse_velocities(particles, primitive, particles_index)

        return primitive

    def gradient_to_ghost(self, particles, gradx, grady, particles_index):
        """
        copy gradient values to ghost particles from their correponding real particles
        """

        gradx, grady = super(Reflect, self).gradient_to_ghost(particles, gradx, grady, particles_index)

        ghost_indices = particles_index["ghost"]

        # reverse velocities in x direction
        x = particles[0,ghost_indices]
        i = np.where((x < self.left) | (self.right < x))[0]
        gradx[1, ghost_indices[i]] *= -1.0
        #grady[1, ghost_indices[i]] *= -1.0

        # reverse velocities in y direction
        y = particles[1,ghost_indices]
        i = np.where((y < self.bottom) | (self.top < y))[0]
        gradx[2, ghost_indices[i]] *= -1.0
        #grady[2, ghost_indices[i]] *= -1.0

        return gradx, grady
