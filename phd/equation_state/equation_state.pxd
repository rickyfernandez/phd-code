from ..containers.containers cimport CarrayContainer

cdef class EquationStateBase:
    cdef public double param_gama

    cpdef conserative_from_primitive(self, CarrayContainer particles)
    cpdef primitive_from_conserative(self, CarrayContainer particles)

    cpdef sound_speed(self, CarrayContainer particles)

    cpdef pressure_by_index(self, CarrayContainer particles)
    cpdef energy_by_index(self, CarrayContainer particles)
    cpdef soundspeed_by_index(self, CarrayContainer particles)

cdef class IdealGas(EquationStateBase):
    pass
