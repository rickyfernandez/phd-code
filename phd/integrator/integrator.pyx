"""Implementation for the integrator"""

from utils.carray cimport BaseArray, DoubleArray, IntArray, LongArray, LongLongArray
from particles.particle_tags import ParticleTAGS
from particles.particle_array import ParticleArray

cdef int Real = ParticleTAGS.Real
cdef int Boundary = ParticleTAGS.Boundary


cdef class Integrator:
    def __init__(self, RightHandSide rhs, int regularize, double eta = 0.25):
        """Constructor for the Integrator"""
        self.rhs = rhs

        self.regularize = regularize
        self.eta = eta

        # properties inherited from the function
        self.mesh = rhs.mesh
        self.pa = rhs.pa

        # create flux data array
        flux_vars = {
                "mass": "double",
                "momentum-x": "double",
                "momentum-y": "double",
                "energy": "double",
                }
        self.fluxes = ParticleArray(flux_vars)

        # create left/right face state array 
        state_vars = {
                "density": "double",
                "velocity-x": "double",
                "velocity-y": "double",
                "pressure": "double",
                }
        self.left_state = ParticleArray(state_vars)
        self.right_state = ParticleArray(state_vars)

    def integrate(self, double dt, double t, int iteration_count):
        self._integrate(dt, t, iteration_count)

    cdef _integrate(self, double dt, double t, int iteration_count):
        """Main step routine"""

        cdef LongArray pair_i = faces.get_carray("pair-i")
        cdef LongArray pair_j = faces.get_carray("pair-j")
        cdef DoubleArray area = faces.get_carray("area")

        cdef DoubleArray m  = self.pa.get_carray("mass")
        cdef DoubleArray mu = self.pa.get_carray("momentum-x")
        cdef DoubleArray mv = self.pa.get_carray("momentum-y")
        cdef DoubleArray E  = self.pa.get_carray("energy")

        cdef DoubleArray f_m  = self.flux.get_carray("mass")
        cdef DoubleArray f_mu = self.flux.get_carray("momentum-x")
        cdef DoubleArray f_mv = self.flux.get_carray("momentum-y")
        cdef DoubleArray f_E  = self.flux.get_carray("energy")

        cdef int i, j, k
        cdef double a

        cdef long num_faces = faces.get_number_of_particles()
        cdef long npart = self.pa.get_number_of_particles()


        # compute particle and face velocities
        self.compute_face_velocities()

        # resize face/states arrays
        self.left_state(num_faces)
        self.right_state(num_faces)
        self.fluxes(num_faces)

        # reconstruct left\right states at each face
        self.rhs.interpolation.compute(self.pa, mesh.faces, self.left_state, self.right_state,
                t, dt, iteration_count)

        # extrapolate state to face, apply frame transformations, solve riemann solver, and transform back
        self.rhs.solve(self.fluxes, self.left_state, self.right_state, mesh.faces,
                t, dt, iteration_count)

        # update conserved quantities
        for k in range(num_faces):

            # update only real particles in the domain
            i = pair_i.data[k]
            j = pair_j.data[k]
            a = area.data[k]

            # flux entering cell defined by particle i
            if tags.data[j] == Real:
                m.data[i]  -= dt*a*f_m.data[k]
                mu.data[i] -= dt*a*f_mu.data[k]
                mv.data[i] -= dt*a*f_mv.data[k]
                E.data[i]  -= dt*a*f_E.data[k]

            # flux leaving cell defined by particle j
            if tags.data[j] == Real:
                m.data[j]  += dt*a*f_m.data[k]
                mu.data[j] += dt*a*f_mu.data[k]
                mv.data[j] += dt*a*f_mv.data[k]
                E.data[j]  += dt*a*f_E.data[k]

        # move particles
        for i in range(npart):
            if tags.data[i] == Real:
                x.data[i] += self.dt*wx.data[i]
                y.data[i] += self.dt*wy.data[i]

    cdef _compute_face_velocities(self):
        self.assign_particle_velocities()
        self.assign_face_velocities(faces)

    cdef _assign_particle_velocities(self):
        """
        Assigns particle velocities. Particle velocities are
        equal to local fluid velocity plus a regularization
        term. The algorithm is taken from Springel (2009).
        """
        # particle flag information
        cdef IntArray type = self.pa.get_carray("type")
        cdef IntArray tags = self.pa.get_carray("tags")

        # particle position
        cdef DoubleArray x = self.pa.get_carray("position-x")
        cdef DoubleArray y = self.pa.get_carray("position-y")

        # center of mass of particle cell
        cdef DoubleArray cx = self.pa.get_carray("com-x")
        cdef DoubleArray cy = self.pa.get_carray("com-y")

        # particle values
        cdef DoubleArray r = self.pa.get_carray("density")
        cdef DoubleArray p = self.pa.get_carray("pressure")
        cdef DoubleArray u = self.pa.get_carray("velocity-x")
        cdef DoubleArray v = self.pa.get_carray("velocity-y")

        # local variables
        cdef double _x, _y, _cx, _cy, _wx, _wy, cs, d, R
        cdef double eta = self.eta

        cdef long npart = self.pa.get_number_of_particles()

        for i in range(npart):
            if tags.data[i] == Real or type.data[i] == Boundary:

                _wx = _wy = 0.0

                if self.regularize == 1:

                    # sound speed 
                    cs = np.sqrt(self.gamma*p.data[i]/r.data[i])

                    # particle positions and center of mass of real particles
                    _x = x.data[i]
                    _y = y.data[i]

                    _cx = cx.data[i]
                    _cy = cy.data[i]

                    # distance form cell com to particle position
                    d = np.sqrt( (_cx - _x)**2 + (_cy - _y)**2 )

                    # approximate length of cell
                    R = np.sqrt(v.data[i]/np.pi)

                    # regularize - eq. 63
                    if ((0.9 <= d/(eta*R)) and (d/(eta*R) < 1.1)):
                        _wx += cs*(_cx - _x)*(d - 0.9*eta*R)/(d*0.2*eta*R)
                        _wy += cs*(_cy - _y)*(d - 0.9*eta*R)/(d*0.2*eta*R)

                    elif (1.1 <= d/(eta*R)):
                        _wx += cs*(_cx - _x)/d
                        _wy += cs*(_cy - _y)/d

                # add velocity of the particle
                wx.data[i] = _wx + u.data[i]
                wy.data[i] = _wy + v.data[i]


    cdef assign_face_velocities(self, ParticleArray faces):
        """
        Assigns velocities to the center of mass of the face
        defined by neighboring particles. The face velocity
        is the average of particle velocities that define
        the face plus a residual motion. The algorithm is
        taken from Springel (2009).
        """
        # particle flag information
        cdef IntArray tag = self.pa.get_carray("tag")
        cdef IntArray type = self.pa.get_carray("type")

        # particle position
        cdef DoubleArray x = self.pa.get_carray("position-x")
        cdef DoubleArray y = self.pa.get_carray("position-y")

        # particle velocity
        cdef DoubleArray wx = self.pa.get_carray("w-x")
        cdef DoubleArray wy = self.pa.get_carray("w-y")

        # face information
        cdef DoubleArray fu  = self.faces.get_carray("velocity-x")
        cdef DoubleArray fv  = self.faces.get_carray("velocity-y")
        cdef DoubleArray fcx = self.faces.get_carray("com-x")
        cdef DoubleArray fcy = self.faces.get_carray("com-y")

        # neighbor arrays
        cdef np.int32[:] neighbors = self.mesh["neighbors"]
        cdef np.int32[:] num_neighbors = self.mesh["number of neighbors"]

        # local variables
        cdef double _xi, _yi, _xj, _yj
        cdef double _wxi, _wyi, _wxj, _wyj
        cdef double factor

        # k is the face index and ind is the neighbor index
        cdef int k = ind = 0
        cdef int n

        cdef long npart = self.pa.get_number_of_particles()

        for i in range(npart):
            if tags.data[i] == Real or type.data[i] == Boundary:

                # particle position
                _xi = x.data[i]
                _yi = y.data[i]

                # particle velocity
                _wxi = wx.data[i]
                _wyi = wy.data[i]

                # assign velocities to faces of particle i
                # by looping over all it's neighbors
                for n in range(num_neighbors[i]):

                    # index of neighbor
                    j = neighbors[ind]

                    # no duplicate faces
                    if i < j:

                        # position of neighbor
                        _xj = x.data[j]
                        _yj = y.data[j]

                        # velocity of neighbor
                        _wxj = wx.data[j]
                        _wyj = wy.data[j]

                        # correct face velocity due to residual motion - eq. 32
                        factor  = (_wxi - _wxj)*(fcx.data[k] - 0.5*(_xi + _xj)) +\
                                  (_wyi - _wyj)*(fcy.data[k] - 0.5*(_yi + _yj))
                        factor /= (_xj - _xi)*(_xj - _xi) + (_yj - _yi)*(_yj - _yi)

                        # the face velocity mean of particle velocities and residual term - eq. 33
                        fu.data[k] = 0.5*(_wxi + _wxj) + factor*(_xj - _xi)
                        fv.data[k] = 0.5*(_wxi + _wxj) + factor*(_yj - _yi)

                        # update counter
                        k   += 1
                        ind += 1

                    else:

                        # face accounted for, go to next neighbor and face
                        ind += 1
