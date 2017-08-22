
cdef class EquationStateBase:
    '''
    Equation of state base. All equation of states must inherit this
    class.
    '''
    def __init__(self, param_gamma = 1.4):
        self.param_gamma = param_gamma

    cpdef conserative_from_primitive(self, CarrayContainer particles):
        '''
        Computes conserative variables from primitive variables
        '''
        msg = "EquationStateBase::conserative_from_primitive called!"
        raise NotImplementedError(msg)

    cpdef primitive_from_conserative(self, CarrayContainer particles):
        '''
        Computes primitive variables from conserative variables
        '''
        msg = "EquationStateBase::primitive_from_conserative called!"
        raise NotImplementedError(msg)

    cpdef sound_speed(self, CarrayContainer particles):
        msg = "EquationStateBase::sound_speed called!"
        raise NotImplementedError(msg)

    cpdef pressure_by_index(self, CarrayContainer particles):
        msg = "EquationStateBase::pressure_by_index called!"
        raise NotImplementedError(msg)

    cpdef energy_by_index(self, CarrayContainer particles):
        msg = "EquationStateBase::energy_by_index called!"
        raise NotImplementedError(msg)

    cpdef soundspeed_by_index(self, CarrayContainer particles):
        msg = "EquationStateBase::soundspeed_by_index called!"
        raise NotImplementedError(msg)

cdef class IdealGas(EquationStateBase):
    cpdef conserative_from_primitive(self, CarrayContainer particles):
        '''
        Computes conserative variables from primitive variables
        '''
        cdef DoubleArray m   = particles.get_carray("mass")
        cdef DoubleArray r   = particles.get_carray("density")
        cdef DoubleArray e   = particles.get_carray("energy")
        cdef DoubleArray p   = particles.get_carray("pressure")
        cdef DoubleArray vol = particles.get_carray("volume")

        cdef int i, k, dim
        cdef np.float64_t vs_sq
        cdef np.float64_t *v[3], *mv[3]

        dim = particles.info["dim"]
        particles.pointer_groups(v,  particles.named_groups['velocity'])
        particles.pointer_groups(mv, particles.named_groups['momentum'])

        for i in range(particles.get_number_of_items()):

            # total mass in cell
            m.data[i] = r.data[i]*vol.data[i]

            # total momentum in cell
            v_sq = 0.
            for k in range(dim):
                mv[k][i] = v[k][i]*m.data[i]
                v_sq    += v[k][i]*v[k][i]

            # total energy in cell
            e.data[i] = (.5*r.data[i]*v_sq + p.data[i]/(self.param_gamma-1.))*vol.data[i]

    cpdef primitive_from_conserative(self, CarrayContainer particles):
        '''
        Computes primitive variables from conserative variables
        '''
        cdef DoubleArray m   = particles.get_carray("mass")
        cdef DoubleArray r   = particles.get_carray("density")
        cdef DoubleArray e   = particles.get_carray("energy")
        cdef DoubleArray p   = particles.get_carray("pressure")
        cdef DoubleArray vol = particles.get_carray("volume")

        cdef int i, k
        cdef np.float64_t vs_sq
        cdef np.float64_t *v[3], *mv[3]

        dim = particles.info["dim"]
        particles.pointer_groups(v,  particles.named_groups['velocity'])
        particles.pointer_groups(mv, particles.named_groups['momentum'])

        for i in range(particles.get_number_of_items()):

            # density in cell
            r.data[i] = m.data[i]/vol.data[i]

            # velocity in cell
            v_sq = 0.
            for k in range(dim):
                v[k][i] = mv[k][i]/m.data[i]
                v_sq   += v[k][i]*v[k][i]

            # pressure in cell
            p.data[i] = (e.data[i]/vol.data[i] - .5*r.data[i]*v_sq)*(self.param_gamma-1.)

    cpdef sound_speed(self, CarrayContainer particles):
        '''
        Compute sound speed
        '''
        cdef DoubleArray r = particles.get_carray("density")
        cdef DoubleArray p = particles.get_carray("pressure")
        cdef DoubleArray c = particles.get_carray("sound speed")

        # sound speed 
        for i in range(particles.get_number_of_items()):
            c.data[i] = sqrt(self.param_gamma*p.data[i]/r.data[i])
