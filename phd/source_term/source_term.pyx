
cdef class MUSCLHancockSourceTerm:

    cpdef apply_motion(self, object integrator):
        """Apply terms to particle motion."""
        msg = "MUSCLHancockSourceTerm::apply_motion called!"
        raise NotImplementedError(msg)

    cpdef apply_primitive(self, object integrator):
        """Apply terms to primitive values at face."""
        msg = "MUSCLHancockSourceTerm::apply_primitive called!"
        raise NotImplementedError(msg)

    cpdef apply_conservative(self, object integrator):
        """Apply terms to conservative values after flux update."""
        msg = "MUSCLHancockSourceTerm::apply_conservative called!"
        raise NotImplementedError(msg)

    cpdef apply_flux(self, object integrator):
        """Apply terms using flux variables."""
        msg = "MUSCLHancockSourceTerm::apply_primitive called!"
        raise NotImplementedError(msg)

    cpdef compute_source(self, object integrator):
        """Compute soure term."""
        msg = "MUSCLHancockSourceTerm::compute_source called!"
        raise NotImplementedError(msg)

    cpdef compute_time_step(self, object integrator):
        """Compute time step."""
        msg = "MUSCLHancockSourceTerm::compute_time_step called!"
        raise NotImplementedError(msg)
