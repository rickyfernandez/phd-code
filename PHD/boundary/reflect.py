import numpy as np
from boundary_base import BoundaryBase

class Reflect(BoundaryBase):
    """
    refect boundary class
    """

    def update_boundaries(self, particles, particles_index, neighbor_graph, neighbor_graph_size):
        """
        create ghost particles from real particles using reflective boundary conditins
        """

        ghost_indices = particles_index["ghost"]

        # position of ghost and real particles
        xg = particles[0,ghost_indices]; yg = particles[1,ghost_indices]
        x  = particles[0,:];              y = particles[1,:]

        # arrays for position of new ghost particles
        x_ghost = np.empty(0)
        y_ghost = np.empty(0)

        # hold all the indices of real particles used to create ghost particles
        ghost_mapping_indices = np.empty(0, dtype=np.int32)

        # grab all neighbors of ghost particles, this includes border cells and 
        # neighbors of border cells, then remove ghost particles leaving two layers

        # right boundary
        i = np.where(self.right < xg)[0]
        right_border = self.find_boundary_particles(neighbor_graph, neighbor_graph_size, ghost_indices[i], ghost_indices)

        # reflect particles across right boundary
        x_right_ghost = 2.0*self.right - x[right_border]
        y_right_ghost = y[right_border]

        # add new ghost particles
        x_ghost = np.append(x_ghost, x_right_ghost)
        y_ghost = np.append(y_ghost, y_right_ghost)

        # add real particles indices correspoinding to ghost particles 
        ghost_mapping_indices = np.append(ghost_mapping_indices, right_border)


        # left boundary 
        i = np.where(xg < self.left)[0]
        left_border = self.find_boundary_particles(neighbor_graph, neighbor_graph_size, ghost_indices[i], ghost_indices)

        # reflect particles across left boundary
        x_left_ghost = 2.*self.left - x[left_border]
        y_left_ghost = y[left_border]

        # add new ghost particles
        x_ghost = np.append(x_ghost, x_left_ghost)
        y_ghost = np.append(y_ghost, y_left_ghost)

        # add real particles indices correspoinding to ghost particles 
        ghost_mapping_indices = np.append(ghost_mapping_indices, left_border)


        # top boundary 
        i = np.where(yg > self.top)[0]
        top_border = self.find_boundary_particles(neighbor_graph, neighbor_graph_size, ghost_indices[i], ghost_indices)

        # reflect particles across top boundary
        x_top_ghost = x[top_border]
        y_top_ghost = 2.*self.top - y[top_border]

        # add new ghost particles
        x_ghost = np.append(x_ghost, x_top_ghost)
        y_ghost = np.append(y_ghost, y_top_ghost)

        # add real particles indices correspoinding to ghost particles 
        ghost_mapping_indices = np.append(ghost_mapping_indices, top_border)


        # bottom boundary 
        i = np.where(yg < self.bottom)[0]
        bottom_border = self.find_boundary_particles(neighbor_graph, neighbor_graph_size, ghost_indices[i], ghost_indices)

        # reflect particles across bottom boundary
        x_bottom_ghost = x[bottom_border]
        y_bottom_ghost = 2.*self.bottom - y[bottom_border]

        # reflect particles to the left boundary
        x_ghost = np.append(x_ghost, x_bottom_ghost)
        y_ghost = np.append(y_ghost, y_bottom_ghost)

        # add real particles indices correspoinding to ghost particles 
        ghost_mapping_indices = np.append(ghost_mapping_indices, bottom_border)


        # create the new list of particles
        real_indices = particles_index["real"]

        # first the real particles
        x_new_particles = np.copy(particles[0,real_indices])
        y_new_particles = np.copy(particles[1,real_indices])

        # now the new ghost particles
        x_new_particles = np.append(x_new_particles, x_ghost)
        y_new_particles = np.append(y_new_particles, y_ghost)

        # update particle information, generate new ghost map
        ghost_map = {}
        for i,j in enumerate(np.arange(real_indices.size, x_new_particles.size)):
            ghost_map[j] = ghost_mapping_indices[i]

        # do i really want to reorganize the data?
        particles_index["ghost_map"] = ghost_map
        particles_index["ghost"] = np.arange(real_indices.size, x_new_particles.size)

        return  np.array([x_new_particles, y_new_particles])

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
        #gradx[1, ghost_indices[i]] *= -1.0
        #grady[1, ghost_indices[i]] *= -1.0

        # reverse velocities in y direction
        y = particles[1,ghost_indices]
        i = np.where((y < self.bottom) | (self.top < y))[0]
        #gradx[2, ghost_indices[i]] *= -1.0
        #grady[2, ghost_indices[i]] *= -1.0

        return gradx, grady
