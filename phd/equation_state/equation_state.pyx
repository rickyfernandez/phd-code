from libc.math cimport sqrt

from ..utils.carray cimport DoubleArray

cdef class EquationStateBase:
    '''
    Equation of state base. All equation of states must inherit this
    class.
    '''
    cpdef conservative_from_primitive(self, CarrayContainer particles):
        '''
        Computes conservative variables from primitive variables
        '''
        msg = "EquationStateBase::conservative_from_primitive called!"
        raise NotImplementedError(msg)

    cpdef primitive_from_conservative(self, CarrayContainer particles):
        '''
        Computes primitive variables from conservative variables
        '''
        msg = "EquationStateBase::primitive_from_conservative called!"
        raise NotImplementedError(msg)

    cpdef np.float64_t sound_speed(self, np.float64_t density, np.float64_t pressure):
        msg = "EquationStateBase::pressure_by_index called!"
        raise NotImplementedError(msg)

    cpdef np.float64_t get_gamma(self):
        msg = "EquationStateBase::get_gamma called!"
        raise NotImplementedError(msg)

cdef class IdealGas(EquationStateBase):
    def __init__(self, param_gamma = 1.4):
        self.param_gamma = param_gamma

    cpdef conservative_from_primitive(self, CarrayContainer particles):
        '''
        Computes conservative variables from primitive variables
        '''
        # conservative variables
        cdef DoubleArray m = particles.get_carray("mass")
        cdef DoubleArray e = particles.get_carray("energy")

        # primitive variable
        cdef DoubleArray d = particles.get_carray("density")
        cdef DoubleArray p = particles.get_carray("pressure")

        # particle volume
        cdef DoubleArray vol = particles.get_carray("volume")

        cdef int i, k, dim
        cdef np.float64_t vs_sq
        cdef np.float64_t *v[3], *mv[3]

        dim = len(particles.named_groups['position'])
        particles.pointer_groups(v,  particles.named_groups['velocity'])
        particles.pointer_groups(mv, particles.named_groups['momentum'])

        # loop through all particles (real + ghost)
        for i in range(particles.get_number_of_items()):

            # total mass in cell
            m.data[i] = d.data[i]*vol.data[i]

            # total momentum in cell
            v_sq = 0.
            for k in range(dim):
                mv[k][i] = v[k][i]*m.data[i]
                v_sq    += v[k][i]*v[k][i]

            # total energy in cell
            e.data[i] = (.5*d.data[i]*v_sq + p.data[i]/(self.param_gamma-1.))*vol.data[i]

    cpdef primitive_from_conservative(self, CarrayContainer particles):
        '''
        Computes primitive variables from conservative variables. Calculates
        for all particles (real + ghost).
        '''
        # conservative variables
        cdef DoubleArray m = particles.get_carray("mass")
        cdef DoubleArray e = particles.get_carray("energy")

        # primitive variables
        cdef DoubleArray d = particles.get_carray("density")
        cdef DoubleArray p = particles.get_carray("pressure")

        # particle volume
        cdef DoubleArray vol = particles.get_carray("volume")

        cdef int i, k
        cdef np.float64_t vs_sq
        cdef np.float64_t *v[3], *mv[3]

        dim = len(particles.named_groups['position'])
        particles.pointer_groups(v,  particles.named_groups['velocity'])
        particles.pointer_groups(mv, particles.named_groups['momentum'])

        # loop through all particles (real + ghost)
        for i in range(particles.get_number_of_items()):

            # density in cell
            d.data[i] = m.data[i]/vol.data[i]

            # velocity in cell
            v_sq = 0.
            for k in range(dim):
                v[k][i] = mv[k][i]/m.data[i]
                v_sq   += v[k][i]*v[k][i]

            # pressure in cell
            p.data[i] = (e.data[i]/vol.data[i] - .5*d.data[i]*v_sq)*(self.param_gamma-1.)

    cpdef np.float64_t sound_speed(self, np.float64_t density, np.float64_t pressure):
        """
        Sound speed of particle
        """
        return sqrt(self.param_gamma*pressure/density)

    cpdef np.float64_t get_gamma(self):
        return self.param_gamma
