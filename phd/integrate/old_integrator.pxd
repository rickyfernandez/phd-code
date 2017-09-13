from ..mesh.mesh cimport Mesh
from ..riemann.riemann cimport RiemannBase
from ..containers.containers cimport CarrayContainer

cdef class IntegrateBase:

    cdef public CarrayContainer pc
    cdef public CarrayContainer left_state
    cdef public CarrayContainer right_state
    cdef public CarrayContainer flux

    cdef public double gamma

    cdef public RiemannBase riemann
    cdef public Mesh mesh
    cdef public int dim

    cdef _integrate(self, double dt, double t, int iteration_count)

cdef class MovingMesh(IntegrateBase):

    # particle properties used for time stepping
    cdef int regularize
    cdef double eta

    cdef _integrate(self, double dt, double t, int iteration_count)
    cdef _compute_face_velocities(self)
    cdef _assign_particle_velocities(self)
    cdef _assign_face_velocities(self)
