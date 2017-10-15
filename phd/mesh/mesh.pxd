from libcpp.vector cimport vector

from ..mesh.pytess cimport PyTess
from ..riemann.riemann cimport RiemannBase
from ..domain.domain_manager cimport DomainManager
from ..containers.containers cimport CarrayContainer

ctypedef vector[int] nn           # nearest neighbors
ctypedef vector[nn] nn_vec

cdef class Mesh:

    # initialization parameters
    cdef public int param_dim
    cdef public double param_eta
    cdef public bint param_regularize
    cdef public int param_num_neighbors

    cdef bint particle_fields_registered

    cdef public CarrayContainer faces

    cdef PyTess tess
    cdef nn_vec neighbors

    # mesh generation routines
    cpdef reset_mesh(self)
    cpdef tessellate(self, CarrayContainer pc, DomainManager domain_manager)
    cpdef build_geometry(self, CarrayContainer pc, DomainManager domain_manager)

    cdef assign_generator_velocities(self, CarrayContainer particles)
    cdef assign_face_velocities(self, CarrayContainer particles)
    cdef update_from_fluxes(self, CarrayContainer particles, RiemannBase riemann, double dt)
