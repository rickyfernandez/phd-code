import numpy as np

def reverse_velocities_boundary(particles, primitive, particles_index, boundary_dic):

    # grab domain values
    left = boundary_dic["left"]
    right = boundary_dic["right"]

    bottom = boundary_dic["bottom"]
    top = boundary_dic["top"]

    ghost_indices = particles_index["ghost"]

    # reverse velocities in x direction
    x = particles[ghost_indices, 0]
    i = np.where((x < left) | (right < x))[0]
    primitive[1, ghost_indices[i]] *= -1.0

    # reverse velocities in y direction
    y = particles[ghost_indices, 1]
    i = np.where((y < bottom) | (top < y))[0]
    primitive[2, ghost_indices[i]] *= -1.0
