import numpy as np
from reflect_2d import Reflect2D

class Reflect3D(Reflect2D):
    """
    refect boundary class
    """
    def __init__(self, xl, xr, yl, yr, zl, zr):

        self.dim = 3
        self.boundaries = [
                [xl, xr],   # x dim
                [yl, yr],   # y dim
                [zl, zr]    # z dim
                ]


    def reverse_velocities(self, particles, primitive, particles_index):
        """
        reflect ghost velocities across the mirror axis
        """
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

        # reverse velocities in z direction
        zl = self.boundaries[2][0]
        zr = self.boundaries[2][1]
        z = particles[2,ghost_indices]
        i = np.where((z < zl) | (zr < z))[0]
        primitive[3, ghost_indices[i]] *= -1.0


    def primitive_to_ghost(self, particles, primitive, particles_index):
        """
        copy primitive values to ghost particles from their correponding real particles
        """

        # copy primitive values to ghost
        primitive = super(Reflect3D, self).primitive_to_ghost(particles, primitive, particles_index)

        # ghost particles velocities have to be reversed
        self.reverse_velocities(particles, primitive, particles_index)

        return primitive


    def gradient_to_ghost(self, particles, grad, particles_index):
        """
        copy gradient values to ghost particles from their correponding real particles
        """

        new_grad = super(Reflect3D, self).gradient_to_ghost(particles, grad, particles_index)

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
