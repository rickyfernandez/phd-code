
class ReconstructBase(object):

    def __init__(self, boundary=None):
        self.gradx = None
        self.grady = None
        self.boundary = boundary
