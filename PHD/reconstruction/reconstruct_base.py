
class ReconstructBase(object):
    """
    reconstruction base class, every reconstruction class must inherit
    this class
    """

    def __init__(self, boundary=None):
        self.grad = None
        self.boundary = boundary

    def gradient(self, primitive, particles, particles_index, cells_info, graphs):
        pass

    def extrapolate(self, faces_info, cell_com, gamma, dt):
        pass
