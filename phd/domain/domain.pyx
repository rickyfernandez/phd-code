
cdef class DomainLimits:

    cdef public xmin, xmax
    cdef public ymin, ymax
    cdef public zmin, zmax

    cdef public double xtranslate
    cdef public double ytranslate
    cdef public double ztranslate

    cdef public int dim
    cdef public bint is_periodic
    cdef public bint is_outflow
    cdef public bint is_reflective

    # only square boxes for now
    def __init__(self, int dim=1, double xmin=0, double xmax=1.0,
            #double ymin=0, double ymax=1.0,
            #double zmin=0, double zmax=1.0,
            is_periodic=False, is_outflow=False, is_wall=True):

        self._check_limits(
        self.xmin = xmin; self.xmax = xmax
        self.ymin = xmin; self.ymax = xmax
        self.zmin = xmin; self.zmax = xmax

        # Indicateds if the domain is periodic or outflow
        self.is_periodic = is_periodic
        self.is_outflow = is_outflow
        self.is_wall = is_wall

        # get the translates in each coordinate direction
        self.xtranslate = xmax - xmin
        self.ytranslate = ymax - ymin
        self.ztranslate = zmax - zmin

        # store the dimension
        self.dim = dim

    #def _check_limits(self, xmin, xmax, ymin, zmin, zmax):
    def _check_limits(self, xmin, xmax):
        """Sanity check on the limits."""
        #if ((xmax < xmin)) or (ymax < ymin) or (zmax < zmin)):
        if (xmax < xmin):
            raise ValueError("Invalied domain limits!")
