from libcpp.vector cimport vector

from .pytess cimport PyTess
from ..domain_manager cimport DomainManager
from ..containers.containers cimport CarrayContainer

ctypedef vector[int] nn           # nearest neighbors
ctypedef vector[nn] nn_vec

cdef class MeshBase:

    # initialization parameters
    cdef public double param_eta
    cdef public int param_num_neigh
    cdef public bint param_regularize

    cdef public CarrayContainer faces
    cdef public DomainManager domain_manager

    cdef PyTess tess
    cdef nn_vec neighbors

    # mesh generation routines
    cpdef reset_mesh(self)
    cpdef tessellate(self, CarrayContainer pc)
    cpdef build_geometry(self, CarrayContainer pc)

    cdef assign_generator_velocities(self, CarrayContainer particles)
    cdef assign_face_velocities(self, CarrayContainer particles)
    cdef update_from_fluxes(self, CarrayContainer particles, RiemannBase riemann, double dt)
