
cdef class MUSCLHancockSourceTerm:

    cpdef apply_primitive(self, object integrator)
    cpdef apply_conservative(self, object integrator)
    cpdef apply_flux(self, object integrator)
    cpdef compute_source(self, object integrator)
    cpdef compute_time_step(self, object integrator)
