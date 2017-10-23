import numpy as np

cdef class DomainLimits:
    """
    Physical dimensions of the simulation space
    """
    def __init__(self, np.ndarray[np.float64_t, ndim=1] xmin,
            np.ndarray[np.float64_t, ndim=1] xmax, int dim=2):
        #self._check_limits(xmin, xmax)

        # only square boxes for now
        #self.xmin = xmin
        #self.xmax = xmax

        self.min_length = 0.
        self.max_length = 0.
        for i in range(dim):
            self.bounds[0][i] = xmin[i]
            self.bounds[1][i] = xmax[i]
            self.translate[i] = xmax[i] - xmin[i]
            #print 'translate', self.translate[i]

            self.min_length = min(self.min_length, self.translate[i])
            self.max_length = max(self.max_length, self.translate[i])

        # store the dimension
        self.dim = dim

    def __getitem__(self, tup):
        row, col = tup
        return self.bounds[row][col]

    cdef _check_limits(self, xmin, xmax):
        """Sanity check on the limits."""
        if (xmax < xmin):
            raise ValueError("Invalid domain limits!")
