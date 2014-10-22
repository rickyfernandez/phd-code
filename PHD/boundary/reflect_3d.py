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
        pass


    def primitive_to_ghost(self, particles, primitive, particles_index):
        """
        copy primitive values to ghost particles from their correponding real particles
        """
        pass


    def gradient_to_ghost(self, particles, grad, particles_index):
        """
        copy gradient values to ghost particles from their correponding real particles
        """
        pass
