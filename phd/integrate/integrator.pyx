from ..utils.particle_tags import ParticleTAGS

from ..mesh.mesh cimport Mesh
from ..riemann.riemann cimport RiemannBase
from ..containers.containers cimport CarrayContainer, ParticleContainer
from ..utils.carray cimport DoubleArray, IntArray, LongLongArray, LongArray
from libc.math cimport sqrt, fabs, fmin, pow

import numpy as np
cimport numpy as np

cdef int Real = ParticleTAGS.Real

cdef class IntegrateBase:
    def __init__(self, ParticleContainer pc, Mesh mesh, RiemannBase riemann):
        """Constructor for the Integrator"""

        self.dim = mesh.dim
        self.mesh = mesh
        self.riemann = riemann
        self.particles = pc

        self.gamma = riemann.gamma

        # create flux data array
        flux_vars = {
                "mass": "double",
                "momentum-x": "double",
                "momentum-y": "double",
                "energy": "double",
                }

        if self.dim == 3:
            flux_vars["momentum-z"] = "double"

        self.flux = CarrayContainer(var_dict=flux_vars)

        # create left/right face state array 
        state_vars = {
                "density": "double",
                "velocity-x": "double",
                "velocity-y": "double",
                "pressure": "double",
                }

        if self.dim == 3:
            state_vars["velocity-z"] = "double"

        self.left_state  = CarrayContainer(var_dict=state_vars)
        self.right_state = CarrayContainer(var_dict=state_vars)

    def compute_time_step(self):
        return self._compute_time_step()

    def integrate(self, double dt, double t, int iteration_count):
        self._integrate(dt, t, iteration_count)

    cdef double _compute_time_step(self):
        msg = "IntegrateBase::compute_time_step called!"
        raise NotImplementedError(msg)

    cdef _integrate(self, double dt, double t, int iteration_count):
        msg = "IntegrateBase::_integrate called!"
        raise NotImplementedError(msg)


cdef class MovingMesh(IntegrateBase):
    def __init__(self, ParticleContainer pc, Mesh mesh, RiemannBase riemann, int regularize = 0, double eta = 0.25):
        """Constructor for the Integrator"""

        IntegrateBase.__init__(self, pc, mesh, riemann)

        self.regularize = regularize
        self.eta = eta


    cdef _integrate(self, double dt, double t, int iteration_count):
        """Main step routine"""

        # particle flag information
        cdef IntArray tags = self.particles.get_carray("tag")

        # face information
        cdef LongArray pair_i = self.mesh.faces.get_carray("pair-i")
        cdef LongArray pair_j = self.mesh.faces.get_carray("pair-j")
        cdef DoubleArray area = self.mesh.faces.get_carray("area")

        # particle values
        cdef DoubleArray m  = self.particles.get_carray("mass")
        cdef DoubleArray e  = self.particles.get_carray("energy")

        # flux values
        cdef DoubleArray fm  = self.flux.get_carray("mass")
        cdef DoubleArray fe  = self.flux.get_carray("energy")

        cdef int i, j, k, n
        cdef double a

        cdef DoubleArray arr
        cdef np.float64_t *x[3], *wx[3], *mv[3], *fmv[3]
        cdef str field, axis

        cdef int num_faces = self.mesh.faces.get_number_of_items()
        cdef int npart = self.particles.get_number_of_particles()


        # compute particle and face velocities
        self._compute_face_velocities()

        # resize face/states arrays
        self.left_state.resize(num_faces)
        self.right_state.resize(num_faces)
        self.flux.resize(num_faces)

        # reconstruct left\right states at each face
        self.riemann.reconstruction.compute(self.particles, self.mesh.faces, self.left_state, self.right_state,
                self.mesh, self.gamma, dt)

        # extrapolate state to face, apply frame transformations, solve riemann solver, and transform back
        self.riemann.solve(self.flux, self.left_state, self.right_state, self.mesh.faces,
                t, dt, iteration_count, self.mesh.dim)

        self.particles.extract_field_vec_ptr(mv, "momentum")
        self.flux.extract_field_vec_ptr(fmv, "momentum")

        # update conserved quantities
        for n in range(num_faces):

            # update only real particles in the domain
            i = pair_i.data[n]
            j = pair_j.data[n]
            a = area.data[n]

            # flux entering cell defined by particle i
            if tags.data[i] == Real:
                m.data[i] -= dt*a*fm.data[n]
                e.data[i] -= dt*a*fe.data[n]

                for k in range(self.dim):
                    mv[k][i] -= dt*a*fmv[k][n]

            # flux leaving cell defined by particle j
            if tags.data[j] == Real:
                m.data[j] += dt*a*fm.data[n]
                e.data[j] += dt*a*fe.data[n]

                for k in range(self.dim):
                    mv[k][j] += dt*a*fmv[k][n]

        self.particles.extract_field_vec_ptr(x, "position")
        self.particles.extract_field_vec_ptr(wx, "w")

        # move particles
        for i in range(npart):
            if tags.data[i] == Real:
                for k in range(self.dim):
                    x[k][i] += dt*wx[k][i]


    cdef double _compute_time_step(self):

        cdef IntArray tags = self.particles.get_carray("tag")
        cdef DoubleArray r = self.particles.get_carray("density")
        cdef DoubleArray p = self.particles.get_carray("pressure")
        cdef DoubleArray vol = self.particles.get_carray("volume")

        cdef double gamma = self.gamma
        cdef double c, R, dt, vi
        cdef int i, k, npart
        cdef np.float64_t* v[3]


        self.particles.extract_field_vec_ptr(v, "velocity")

        c = sqrt(gamma*p.data[0]/r.data[0])
        if self.dim == 2:
            R = sqrt(vol.data[0]/np.pi)
        if self.dim == 3:
            R = pow(3.0*vol.data[0]/(4.0*np.pi), 1.0/3.0)

        vi = 0.0
        for k in range(self.dim):
            vi += v[k][0]*v[k][0]
        dt = R/(c + sqrt(vi))

        for i in range(self.particles.get_number_of_particles()):
            if tags.data[i] == Real:

                c = sqrt(gamma*p.data[i]/r.data[i])

                # calculate approx radius of each voronoi cell
                if self.dim == 2:
                    R = sqrt(vol.data[i]/np.pi)
                if self.dim == 3:
                    R = pow(3.0*vol.data[i]/(4.0*np.pi), 1.0/3.0)

                vi = 0.0
                for k in range(self.dim):
                    vi += v[k][i]*v[k][i]
                dt = fmin(R/(c + sqrt(vi)), dt)

        return dt

    cdef _compute_face_velocities(self):
        self._assign_particle_velocities()
        self._assign_face_velocities()

    cdef _assign_particle_velocities(self):
        """
        Assigns particle velocities. Particle velocities are
        equal to local fluid velocity plus a regularization
        term. The algorithm is taken from Springel (2009).
        """
        # particle flag information
        cdef IntArray tags = self.particles.get_carray("tag")

        # particle values
        cdef DoubleArray r = self.particles.get_carray("density")
        cdef DoubleArray p = self.particles.get_carray("pressure")
        cdef DoubleArray vol = self.particles.get_carray("volume")

        # local variables
        cdef np.float64_t *x[3], *v[3], *wx[3], *dcx[3]
        cdef double cs, d, R
        cdef double eta = self.eta
        cdef int i, k


        self.particles.extract_field_vec_ptr(x, "position")
        self.particles.extract_field_vec_ptr(v, "velocity")
        self.particles.extract_field_vec_ptr(wx, "w")
        self.particles.extract_field_vec_ptr(dcx, "dcom")

        for i in range(self.particles.get_number_of_particles()):

            for k in range(self.dim):
                wx[k][i] = v[k][i]

            if self.regularize == 1:

                # sound speed 
                cs = sqrt(self.gamma*p.data[i]/r.data[i])

                # distance form cell com to particle position
                d = 0.0
                for k in range(self.dim):
                    d += (dcx[k][i])**2
                d = sqrt(d)

                # approximate length of cell
                if self.dim == 2:
                    R = sqrt(vol.data[i]/np.pi)
                if self.dim == 3:
                    R = pow(3.0*vol.data[i]/(4.0*np.pi), 1.0/3.0)

                # regularize - eq. 63
                if ((0.9 <= d/(eta*R)) and (d/(eta*R) < 1.1)):
                    for k in range(self.dim):
                        wx[k][i] += cs*(dcx[k][i])*(d - 0.9*eta*R)/(d*0.2*eta*R)

                elif (1.1 <= d/(eta*R)):
                    for k in range(self.dim):
                        wx[k][i] += cs*(dcx[k][i])/d


    cdef _assign_face_velocities(self):
        """
        Assigns velocities to the center of mass of the face
        defined by neighboring particles. The face velocity
        is the average of particle velocities that define
        the face plus a residual motion. The algorithm is
        taken from Springel (2009).
        """

        # face information
        cdef LongArray pair_i = self.mesh.faces.get_carray("pair-i")
        cdef LongArray pair_j = self.mesh.faces.get_carray("pair-j")

        # local variables
        cdef double factor
        cdef int i, j, k, n
        cdef np.float64_t *x[3], *wx[3], *fv[3], *fcx[3]

        self.particles.extract_field_vec_ptr(x, "position")
        self.particles.extract_field_vec_ptr(wx, "w")

        self.mesh.faces.extract_field_vec_ptr(fv, "velocity")
        self.mesh.faces.extract_field_vec_ptr(fcx, "com")

        for n in range(self.mesh.faces.get_number_of_items()):

            # particles that define face
            i = pair_i.data[n]
            j = pair_j.data[n]

            # correct face velocity due to residual motion - eq. 32
            factor = denom = 0.0
            for k in range(self.dim):
                factor += (wx[k][i] - wx[k][j])*(fcx[k][n] - 0.5*(x[k][i] + x[k][j]))
                denom  += pow(x[k][j] - x[k][i], 2.0)
            factor /= denom

            # the face velocity mean of particle velocities and residual term - eq. 33
            for k in range(self.dim):
                fv[k][n] = 0.5*(wx[k][i] + wx[k][j]) + factor*(x[k][j] - x[k][i])
