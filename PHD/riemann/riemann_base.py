import numpy as np

class riemann_base(object):
    """
    Riemann base class. All riemann solvers should inherit this class
    """

    def __init__(self, smallp=1.0E-10, smallc=1.0E-10, smallrho=1.0E-10):
        """
        Set parameters for riemann class.
        """

        self.small_pressure = smallp
        self.small_sound_speed = smallc
        self.small_rho = smallrho
