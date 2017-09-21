
cdef class DomainLimits:
    """
    Physical dimensions of the simulation space
    """
    def __init__(self, int dim=2, double xmin=0, double xmax=1.0):
        self._check_limits(xmin, xmax)

        # only square boxes for now
        self.xmin = xmin
        self.xmax = xmax

        for i in range(dim):
            self.bounds[0][i] = xmin
            self.bounds[1][i] = xmax
            self.translate[i] = xmax - xmin

        self.max_length = xmax - xmin
        self.min_length = xmax - xmin

        # store the dimension
        self.dim = dim

    def __getitem__(self, tup):
        row, col = tup
        return self.bounds[row][col]

    cdef _check_limits(self, xmin, xmax):
        """Sanity check on the limits."""
        if (xmax < xmin):
            raise ValueError("Invalid domain limits!")
