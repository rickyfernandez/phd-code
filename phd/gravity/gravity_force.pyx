
cdef class ConstantGravity:
    """
    Constant gravity source term. Gravity is applied in only
    direction.
    """
    def __init__(self, dim=2, grav_axis="y", g=-1., **kwargs):
        self.g = g

        if grav_axis == "x":
            self.axis = 0
        elif grav_axis == "y":
            self.axis = 1
        elif grav_axis == "z":
            self.axis = 2
        else:
            raise RuntimeError("ERROR: Unrecognized gravity axis")

    def initialize(self):
        pass

    def update_reconstruction(self, object integrator):
        """
        Add gravity half time step update to primitive variables
        at faces for riemann solver.
        """
        cdef int i, j, m
        cdef double dt = integrator.dt
        cdef np.float64_t *vl[3], *vr[3]
        cdef CarrayContainer left_states = integrator.reconstruction.left_states
        cdef CarrayContainer right_states = integrator.reconstruction.right_states

        left_state.pointer_groups(vl,  left_state.named_groups["velocity"])
        right_state.pointer_groups(vr, left_state.named_groups["velocity"])

        # loop over each face in the mesh 
        for m in range(integrator.mesh.faces.get_carray_size()):

            # extract particles that defined the face
            i = pair_i.data[m]
            j = pair_j.data[m]

            # add gravity to velocity
            vl[self.axis][m] += 0.5*dt*self.g
            vr[self.axis][m] += 0.5*dt*self.g

    def update_conserved(self, object integrator):
        """
        Update conservative variables before integrate.
        """
        cdef int i
        cdef np.float64_t *mv[3]

        cdef IntArray tags    = integrate.particles.get_carray("tag")
        cdef DoubleArray mass = integrator.particles.get_carray("mass")
        cdef DoubleArray e    = integrator.particles.get_carray("energy")

        integrator.particles.pointer_groups(mv,
                integrator.particles.named_groups["momentum"])

        # add gravity acceleration from particle
        for i in range(integrator.particles.get_carray_size()):
            if tags.data[i] == REAL:
                e.data[i] += 0.5*dt*mv[self.axis][i]*g  # energy
                mv[self.axis][i] += 0.5*dt*m.data[i]*g  # momentum

    def update_conserved(self, object integrator):
        """
        Update conservative variables before integrate.
        """
        cdef int i
        cdef np.float64_t *mv[3]

        cdef IntArray tags    = integrate.particles.get_carray("tag")
        cdef DoubleArray mass = integrator.particles.get_carray("mass")
        cdef DoubleArray e    = integrator.particles.get_carray("energy")

        integrator.particles.pointer_groups(mv,
                integrator.particles.named_groups["momentum"])

        # add gravity acceleration from particle
        for i in range(integrator.particles.get_carray_size()):
            if tags.data[i] == REAL:
                mv[self.axis][i] += 0.5*dt*m.data[i]*g  # momentum
                e.data[i] += 0.5*dt*mv[self.axis][i]*g  # energy

#cdef class GravityForce(SourceTermBase):
#
#    def _initialize(self):
#        pass
#
#    def before_mainloop(self):
#        # calculate acceleration and potentail
#        self.gr_tree.walk()
#
#    def after_integrate(self):
#
#        # calculate acceleration and potentail
#        self.gr_tree.walk()
#
#    def after_compute_face_velocities(self, MeshMotion mesh_motion):
#        """
#        Add gravity half time step update to primitive variables
#        at faces for riemann solver.
#        """
#        cdef IntArray tags = self.pc.get_carray('tag')
#
#        cdef int i, j, k, m
#        cdef np.float64_t *mv[3], *a[3]
#
#        # modify velocity what about momentum? is this okay
#        self.pc.pointer_groups(v, self.pc.named_groups['velocity'])
#        self.pc.pointer_groups(a, self.pc.named_groups['acceleration'])
#
#        # kick velocities
#        for i in range(self.pc.get_number_of_items()):
#            if tags.data[i] == Real:
#                for k in range(self.dim):
#                    v[k][i] += 0.5*dt*a[k][i]
#
#    def before_integrate(self, IntegrateBase integrate):
#        """
#        Add half step gravity to the conserative equations.
#        This has to be called twice, first for old acceleration
#        and second for new acceleration.
#        """
#        cdef IntArray tags = self.pc.get_carray('tag')
#
#        # face information
#        cdef LongArray pair_i = self.mesh.faces.get_carray("pair-i")
#        cdef LongArray pair_j = self.mesh.faces.get_carray("pair-j")
#        cdef DoubleArray area = self.mesh.faces.get_carray("area")
#
#        # particle values
#        cdef DoubleArray m  = self.pc.get_carray("mass")
#        cdef DoubleArray e  = self.pc.get_carray("energy")
#
#        cdef DoubleArray rij = self.rij
#
#        # flux values
#        cdef DoubleArray fm  = self.flux.get_carray("mass")
#
#        cdef int i, j, k, m
#        cdef np.float64_t *mv[3], *a[3]
#
#        self.pc.pointer_groups(wx, self.pc.named_groups['w'])
#        self.pc.pointer_groups(mv, self.pc.named_groups['momentum'])
#        self.pc.pointer_groups(mv, self.pc.named_groups['acceleration'])
#
#        # add gravity acceleration from particle 
#        for i in range(self.pc.get_number_of_items()):
#            if tags.data[i] == Real:
#                for k in range(self.dim):
#                    mv[k][i]  -= 0.5*dt*m.data[i]*a[k][i]          # momentum
#                    e.data[i] -= 0.5*dt*m.data[i]*w[k][i]*a[k][i]  # energy
#
#        rij.resize(faces.get_number_of_items())
#
#        # add gravity acceleration from neighbors 
#        for m in range(faces.get_number_of_items()):
#
#            i = pair_i.data[m]
#            j = pair_j.data[m]
#
#            for k in range(self.dim):
#                e.data[i] += dt*a*fm.data[m]
#                e.data[j] -= dt*a*fm.data[m]
#
#                # store separation vector
#                rij.data[m] = x[k][i] - x[k][j]
#
#    def after_mesh(self, MeshBase mesh):
#        """
#        Add half step gravity to the conserative equations.
#        This has to be called twice, first for old acceleration
#        and second for new acceleration.
#        """
#        cdef IntArray tags = self.pc.get_carray('tag')
#
#        # face information
#        cdef LongArray pair_i = self.mesh.faces.get_carray("pair-i")
#        cdef LongArray pair_j = self.mesh.faces.get_carray("pair-j")
#        cdef DoubleArray area = self.mesh.faces.get_carray("area")
#
#        # particle values
#        cdef DoubleArray m  = self.pc.get_carray("mass")
#        cdef DoubleArray e  = self.pc.get_carray("energy")
#
#        # flux values
#        cdef DoubleArray fm  = self.flux.get_carray("mass")
#
#        cdef DoubleArray rij = self.rij
#
#        cdef int i, j, k, m
#        cdef np.float64_t *mv[3], *a[3]
#
#        self.pc.pointer_groups(wx, self.pc.named_groups['w'])
#        self.pc.pointer_groups(mv, self.pc.named_groups['momentum'])
#        self.pc.pointer_groups(mv, self.pc.named_groups['acceleration'])
#
#        # add gravity acceleration from particle 
#        for i in range(self.pc.get_number_of_items()):
#            if tags.data[i] == Real:
#                for k in range(self.dim):
#                    mv[k][i]  -= 0.5*dt*m.data[i]*a[k][i]          # momentum
#                    e.data[i] -= 0.5*dt*m.data[i]*w[k][i]*a[k][i]  # energy
#
#        # add gravity acceleration from neighbors 
#        for m in range(faces.get_number_of_items()):
#
#            i = pair_i.data[m]
#            j = pair_j.data[m]
#
#            # update energy eq. 94
#            for k in range(self.dim):
#                e.data[i] += dt*a*fm.data[m]*rij.data[m]
#                e.data[j] += dt*a*fm.data[m]*rij.data[m]
