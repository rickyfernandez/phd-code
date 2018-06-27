import logging
import numpy as np

cimport numpy as np
from libc.math cimport sqrt, fmin

from ..utils.particle_tags import ParticleTAGS

from ..containers.containers cimport CarrayContainer
from ..utils.carray cimport DoubleArray, IntArray, LongArray

cdef int REAL = ParticleTAGS.Real

phdLogger = logging.getLogger("phd")

cdef class ConstantGravity(MUSCLHancockSourceTerm):
    """
    Constant gravity source term. Gravity is applied in only
    direction.
    """
    def __init__(self, grav_axis="y", g=-1., **kwargs):
        self.g = g

        if grav_axis == "x":
            self.axis = 0
        elif grav_axis == "y":
            self.axis = 1
        elif grav_axis == "z":
            self.axis = 2
        else:
            raise RuntimeError("ERROR: Unrecognized gravity axis")

        self.grav_axis = grav_axis

    cpdef apply_motion(self, object integrator):
        pass

    cpdef apply_primitive(self, object integrator):
        """
        Add gravity half time step update to primitive variables
        at faces for riemann solver and add half update to
        conservative variables.
        """
        cdef int i, j, m

        cdef double g = self.g
        cdef int axis = self.axis
        cdef double dt = integrator.dt

        cdef np.float64_t *vl[3], *vr[3], *wx[3], *mv[3]

        cdef IntArray tags    = integrator.particles.get_carray("tag")
        cdef DoubleArray mass = integrator.particles.get_carray("mass")
        cdef DoubleArray e    = integrator.particles.get_carray("energy")

        cdef LongArray pair_i = integrator.mesh.faces.get_carray("pair-i")
        cdef LongArray pair_j = integrator.mesh.faces.get_carray("pair-j")

        cdef CarrayContainer left_states = integrator.reconstruction.left_states
        cdef CarrayContainer right_states = integrator.reconstruction.right_states

        cdef CarrayContainer particles = integrator.particles

        phdLogger.info("ConstantGravity: Applying gravity to primitive")

        left_states.pointer_groups(vl,  left_states.carray_named_groups["velocity"])
        right_states.pointer_groups(vr, left_states.carray_named_groups["velocity"])

        particles.pointer_groups(mv, particles.carray_named_groups["momentum"])
        particles.pointer_groups(wx, particles.carray_named_groups["w"])

        # loop over each face in the mesh 
        for m in range(integrator.mesh.faces.get_carray_size()):

            # extract particles that defined the face
            i = pair_i.data[m]
            j = pair_j.data[m]

            # add gravity to velocity
            vl[axis][m] += 0.5*dt*g
            vr[axis][m] += 0.5*dt*g

        # add gravity acceleration from particle
        for i in range(integrator.particles.get_carray_size()):
            if tags.data[i] == REAL:
                e.data[i] += 0.5*dt*mv[axis][i]*g     # energy
                #e.data[i] += 0.5*dt*mass.data[i]*wx[axis][i]*g    # energy
                mv[axis][i] += 0.5*dt*mass.data[i]*g  # momentum

    cpdef compute_source(self, object integrator):
        pass

    cpdef compute_time_step(self, object integrator):
        return integrator.dt

    cpdef apply_flux(self, object integrator):
        pass
#        cdef int i, j, n, axis = self.axis
#        cdef double *x[3], g = self.g
#        cdef double a, dt = integrator.dt
#
#        # particle indices that make up face
#        cdef LongArray pair_i = integrator.mesh.faces.get_carray("pair-i")
#        cdef LongArray pair_j = integrator.mesh.faces.get_carray("pair-j")
#        cdef DoubleArray area = integrator.mesh.faces.get_carray("area")
#
#        cdef DoubleArray fm = integrator.riemann.fluxes.get_carray("mass")
#
#        cdef IntArray tags    = integrator.particles.get_carray("tag")
#        cdef DoubleArray e = integrator.particles.get_carray("energy")
#
#        cdef CarrayContainer particles = integrator.particles
#
#        particles.pointer_groups(x, particles.carray_named_groups["position"])
#
#        for n in range(integrator.mesh.faces.get_carray_size()):
#
#            # particles that make up the face
#            i = pair_i.data[n]
#            j = pair_j.data[n]
#
#            # area of the face
#            a = area.data[n]
#
#            # shoud be wrong?
#            if(tags.data[i] == REAL):
#                e.data[i] -= 0.5*dt*a*fm.data[n]*(x[axis][i] - x[axis][j])*g
#                #e.data[i] -= 0.25*dt*a*fm.data[n]*(x[axis][i] - x[axis][j])*g
#
#            if(tags.data[j] == REAL):
#                e.data[j] += 0.5*dt*a*fm.data[n]*(x[axis][j] - x[axis][i])*g
                #e.data[j] += 0.25*dt*a*fm.data[n]*(x[axis][j] - x[axis][i])*g

    cpdef apply_conservative(self, object integrator):
        """Update conservative variables after flux update."""
        cdef int i
        cdef np.float64_t *mv[3]
        cdef np.float64_t *wx[3]

        cdef double g = self.g
        cdef int axis = self.axis
        cdef double dt = integrator.dt

        cdef IntArray tags    = integrator.particles.get_carray("tag")
        cdef DoubleArray mass = integrator.particles.get_carray("mass")
        cdef DoubleArray e    = integrator.particles.get_carray("energy")

        cdef CarrayContainer particles = integrator.particles

        phdLogger.info("ConstantGravity: Applying gravity to conservative")

        particles.pointer_groups(mv, particles.carray_named_groups["momentum"])
        particles.pointer_groups(wx, particles.carray_named_groups["w"])

        # add gravity acceleration from particle
        for i in range(particles.get_carray_size()):
            if tags.data[i] == REAL:
                mv[axis][i] += 0.5*dt*mass.data[i]*g  # momentum
                #e.data[i] += 0.5*dt*mass.data[i]*wx[axis][i]*g  # energy
                e.data[i] += 0.5*dt*mv[axis][i]*g  # energy

cdef class SelfGravity(MUSCLHancockSourceTerm):
    """
    Constant gravity source term. Gravity is applied in only
    direction.
    """
    def __init__(self, str split_type="barnes-hut", double barnes_angle=0.3,
            double smoothing_length = 1.0E-5, int calculate_potential=0,
            int max_buffer_size=256, eta=0.1):

        self.eta = eta
        self.split_type = split_type
        self.barnes_angle = barnes_angle
        self.max_buffer_size = max_buffer_size
        self.smoothing_length = smoothing_length
        self.calculate_potential = calculate_potential

        self.gravity = GravityTree(barnes_angle, smoothing_length,
                calculate_potential, max_buffer_size)

    cpdef apply_motion(self, object integrator):
        pass
#        cdef int i, k
#        cdef np.float64_t *a[3]
#        cdef np.float64_t *v[3]
#        cdef np.float64_t *wx[3]
#
#        cdef double dt = integrator.dt
#        cdef CarrayContainer particles = integrator.particles
#
#        phdLogger.info("SelfGravity: Applying gravity to particle motion")
#        dim = len(particles.carray_named_groups["position"])
#
#        particles.pointer_groups(wx, particles.carray_named_groups["w"])
#        particles.pointer_groups(a,  particles.carray_named_groups["acceleration"])
#
#        # kick mesh generator
#        for i in range(particles.get_carray_size()):
#            for k in range(dim):
#                wx[k][i] += 0.5*dt*a[k][i]

    cpdef apply_primitive(self, object integrator):
        """
        Add gravity half time step update to primitive variables
        at faces for riemann solver and add half update to
        conservative variables.
        """
        cdef int i, j, m, dim
        cdef double dt = integrator.dt

        cdef np.float64_t *vl[3], *vr[3], *wx[3], *mv[3], *a[3]

        cdef IntArray tags    = integrator.particles.get_carray("tag")
        cdef DoubleArray mass = integrator.particles.get_carray("mass")
        cdef DoubleArray e    = integrator.particles.get_carray("energy")

        cdef LongArray pair_i = integrator.mesh.faces.get_carray("pair-i")
        cdef LongArray pair_j = integrator.mesh.faces.get_carray("pair-j")

        cdef CarrayContainer left_states = integrator.reconstruction.left_states
        cdef CarrayContainer right_states = integrator.reconstruction.right_states

        cdef CarrayContainer particles = integrator.particles

        phdLogger.info("SelfGravity: Applying gravity to primitive")
        dim = len(particles.carray_named_groups["position"])

        left_states.pointer_groups(vl,  left_states.carray_named_groups["velocity"])
        right_states.pointer_groups(vr, left_states.carray_named_groups["velocity"])

        particles.pointer_groups(wx, particles.carray_named_groups["w"])
        particles.pointer_groups(mv, particles.carray_named_groups["momentum"])
        particles.pointer_groups(a, particles.carray_named_groups["acceleration"])

        # loop over each face in the mesh 
        for m in range(integrator.mesh.faces.get_carray_size()):

            # extract particles that defined the face
            i = pair_i.data[m]
            j = pair_j.data[m]

            # add gravity to velocity
            for k in range(dim):
                vl[k][m] += 0.5*dt*a[k][i]
                vr[k][m] += 0.5*dt*a[k][j]

        # add gravity acceleration from particle
        for i in range(integrator.particles.get_carray_size()):
            if tags.data[i] == REAL:
                for k in range(dim):

                    # energy Eq. 82
                    #e.data[i] += 0.5*dt*mass.data[i]*wx[k][i]*a[k][i]
                    e.data[i] += 0.5*dt*mv[k][i]*a[k][i]

                    # momentum Eq. 81
                    mv[k][i] += 0.5*dt*mass.data[i]*a[k][i]

    cpdef compute_source(self, object integrator):
        """Compute gravitational acceleration.
        
        Parameters
        ----------
        integrator : IntegrateBase
            Advances the fluid equations by one step.

        """
        phdLogger.info("SelfGravity: Calculating accelerations")
        self.gravity._build_tree(integrator.particles)
        self.gravity.walk(integrator.particles) 

        integrator.domain_manager.update_ghost_fields(
                integrator.particles,
                integrator.particles.carray_named_groups["acceleration"])

    cpdef compute_time_step(self, object integrator):
        """Compute time step from gravitational force.
        
        Parameters
        ----------
        integrator : IntegrateBase
            Advances the fluid equations by one step.

        """
        cdef double *a[3]
        cdef int i, k, dim
        cdef double dt, r, a_mag, eta = self.eta

        cdef CarrayContainer particles = integrator.particles
        cdef IntArray tags = integrator.particles.get_carray("tag")
        cdef DoubleArray vol = integrator.particles.get_carray("volume")

        phdLogger.info("SelfGravity: Compute gravitational time step")
        dim = len(particles.carray_named_groups["position"])

        particles.pointer_groups(a, particles.carray_named_groups["acceleration"])

        # length of cell
        if dim == 2:
            r = sqrt(vol.data[0]/np.pi)
        if dim == 3:
            r = pow(3.0*vol.data[0]/(4.0*np.pi), 1.0/3.0)

        a_mag = 0.
        for k in range(dim):
            a_mag += a[k][0]**2
        a_mag = sqrt(a_mag)

        dt = sqrt(2*1.5*eta*r/a_mag)
        for i in range(particles.get_carray_size()):
            if(tags.data[i] == REAL):

                # approximate length of cell
                if dim == 2:
                    r = sqrt(vol.data[i]/np.pi)
                if dim == 3:
                    r = pow(3.0*vol.data[i]/(4.0*np.pi), 1.0/3.0)

                a_mag = 0.
                for k in range(dim):
                    a_mag += a[k][i]**2
                a_mag = sqrt(a_mag)

            dt = fmin(dt, sqrt(2.*1.5*eta*r/a_mag))

        phdLogger.info("SelfGravity: Gravity dt: %f" %dt)
        return dt

    cpdef apply_flux(self, object integrator):
        """Perform any computation after flux update.
        
        Parameters
        ----------
        integrator : IntegrateBase
            Advances the fluid equations by one step.

        """
        pass
#        cdef int i, j, n, k, dim
#        cdef double *x[3], *a[3]
#        cdef double ar, dt = integrator.dt
#        cdef CarrayContainer particles = integrator.particles
#
#        # particle indices that make up face
#        cdef LongArray pair_i = integrator.mesh.faces.get_carray("pair-i")
#        cdef LongArray pair_j = integrator.mesh.faces.get_carray("pair-j")
#        cdef DoubleArray area = integrator.mesh.faces.get_carray("area")
#
#        cdef IntArray tags = integrator.particles.get_carray("tag")
#        cdef DoubleArray e  = integrator.particles.get_carray("energy")
#        cdef DoubleArray fm = integrator.riemann.fluxes.get_carray("mass")
#
#        dim = len(particles.carray_named_groups["position"])
#
#        particles.pointer_groups(x, particles.carray_named_groups["position"])
#        particles.pointer_groups(a, particles.carray_named_groups["acceleration"])
#
#        for n in range(integrator.mesh.faces.get_carray_size()):
#
#            # particles that make up the face
#            i = pair_i.data[n]
#            j = pair_j.data[n]
#
#            # area of the face
#            ar = area.data[n]
#
#            # Eq. 94
#            if(tags.data[i] == REAL):
#                for k in range(dim):
#                    e.data[i] += 0.5*dt*ar*fm.data[n]*(x[k][i] - x[k][j])*a[k][i]
#
#            if(tags.data[j] == REAL):
#                for k in range(dim):
#                    e.data[j] -= 0.5*dt*ar*fm.data[n]*(x[k][j] - x[k][i])*a[k][j]

    cpdef apply_conservative(self, object integrator):
        """Update conservative variables after mesh build.
        
        Parameters
        ----------
        integrator : IntegrateBase
            Advances the fluid equations by one step.

        """
        cdef int i, k
        cdef np.float64_t *a[3]
        cdef np.float64_t *mv[3]
        cdef np.float64_t *wx[3]

        cdef double dt = integrator.dt

        cdef IntArray tags    = integrator.particles.get_carray("tag")
        cdef DoubleArray mass = integrator.particles.get_carray("mass")
        cdef DoubleArray e    = integrator.particles.get_carray("energy")

        cdef CarrayContainer particles = integrator.particles

        phdLogger.info("SelfGravity: Applying gravity to conservative")
        dim = len(particles.carray_named_groups["position"])

        particles.pointer_groups(wx, particles.carray_named_groups["w"])
        particles.pointer_groups(mv, particles.carray_named_groups["momentum"])
        particles.pointer_groups(a,  particles.carray_named_groups["acceleration"])

        # add gravity acceleration from particle
        for i in range(particles.get_carray_size()):
            if tags.data[i] == REAL:
                for k in range(dim):
                    # momentum Eq. 81
                    mv[k][i] += 0.5*dt*mass.data[i]*a[k][i]

                    # energy Eq. 82
                    #e.data[i] += 0.5*dt*mass.data[i]*wx[k][i]*a[k][i]
                    e.data[i] += 0.5*dt*mv[k][i]*a[k][i]
