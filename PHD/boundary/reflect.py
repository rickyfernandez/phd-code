import numpy as np
from boundary_base import boundary_base

class reflect(boundary_base):

    def __init__(self, left, right, bottom, top):
        self.boundary = {}

        self.boundary["left"] = left
        self.boundary["right"] = right
        self.boundary["bottom"] = bottom
        self.boundary["top"] = top

    def update(self, particles, particles_index, neighbor_graph):

        ghost_indices = particles_index["ghost"]

        xg = particles[ghost_indices,0]; yg = particles[ghost_indices,1]
        x  = particles[:,0];              y = particles[:,1]

        x_ghost = np.empty(0)
        y_ghost = np.empty(0)

        # hold all the indices of real particles used to create ghost particles
        ghost_mapping_indices = np.empty(0, dtype=np.int32)

        # grab all neighbors of ghost particles, this includes border cells and 
        # neighbors of border cells, then remove ghost particles leaving two layers

        # grab domain values
        left   = self.boundary["left"];   right = self.boundary["right"]
        bottom = self.boundary["bottom"]; top   = self.boundary["top"]


        # left boundary
        i = np.where(right < xg)[0]
        right_border = self.find_boundary_particles(neighbor_graph, ghost_indices[i], ghost_indices)

        # reflect particles across right boundary
        x_right_ghost = 2.0*right - x[right_border]
        y_right_ghost = y[right_border]

        # add new ghost particles
        x_ghost = np.append(x_ghost, x_right_ghost)
        y_ghost = np.append(y_ghost, y_right_ghost)

        # add real particles indices correspoinding to ghost particles 
        ghost_mapping_indices = np.append(ghost_mapping_indices, right_border)


        # left boundary 
        i = np.where(xg < left)[0]
        left_border = self.find_boundary_particles(neighbor_graph, ghost_indices[i], ghost_indices)

        # reflect particles across left boundary
        x_left_ghost = 2.*left - x[left_border]
        y_left_ghost = y[left_border]

        # add new ghost particles
        x_ghost = np.append(x_ghost, x_left_ghost)
        y_ghost = np.append(y_ghost, y_left_ghost)

        # add real particles indices correspoinding to ghost particles 
        ghost_mapping_indices = np.append(ghost_mapping_indices, left_border)


        # top boundary 
        i = np.where(yg > top)[0]
        top_border = self.find_boundary_particles(neighbor_graph, ghost_indices[i], ghost_indices)

        # reflect particles across top boundary
        x_top_ghost = x[top_border]
        y_top_ghost = 2.*top - y[top_border]

        # add new ghost particles
        x_ghost = np.append(x_ghost, x_top_ghost)
        y_ghost = np.append(y_ghost, y_top_ghost)

        # add real particles indices correspoinding to ghost particles 
        ghost_mapping_indices = np.append(ghost_mapping_indices, top_border)


        # bottom boundary 
        i = np.where(yg < bottom)[0]
        bottom_border = self.find_boundary_particles(neighbor_graph, ghost_indices[i], ghost_indices)

        # reflect particles across bottom boundary
        x_bottom_ghost = x[bottom_border]
        y_bottom_ghost = 2.*bottom - y[bottom_border]

        # reflect particles to the left boundary
        x_ghost = np.append(x_ghost, x_bottom_ghost)
        y_ghost = np.append(y_ghost, y_bottom_ghost)

        # add real particles indices correspoinding to ghost particles 
        ghost_mapping_indices = np.append(ghost_mapping_indices, bottom_border)


        #---------------------------------------------------------------------------
        # create the new list of particles
        real_indices = particles_index["real"]

        # first the real particles
        x_new_particles = np.copy(particles[real_indices,0])
        y_new_particles = np.copy(particles[real_indices,1])

        # now the new ghost particles
        x_new_particles = np.append(x_new_particles, x_ghost)
        y_new_particles = np.append(y_new_particles, y_ghost)

        # update particle information
        # generate new ghost map
        ghost_map = {}
        for i,j in enumerate(np.arange(real_indices.size, x_new_particles.size)):
            ghost_map[j] = ghost_mapping_indices[i]

        # do i really want to reorganize the data?
        particles_index["ghost_map"] = ghost_map
        particles_index["ghost"] = np.arange(real_indices.size, x_new_particles.size)

        return  np.array(zip(x_new_particles, y_new_particles))

    def reverse_velocities(self, particles, primitive, particles_index):

        # grab domain values
        left = self.boundary["left"]
        right = self.boundary["right"]

        bottom = self.boundary["bottom"]
        top = self.boundary["top"]

        ghost_indices = particles_index["ghost"]

        # reverse velocities in x direction
        x = particles[ghost_indices, 0]
        i = np.where((x < left) | (right < x))[0]
        primitive[1, ghost_indices[i]] *= -1.0

        # reverse velocities in y direction
        y = particles[ghost_indices, 1]
        i = np.where((y < bottom) | (top < y))[0]
        primitive[2, ghost_indices[i]] *= -1.0
