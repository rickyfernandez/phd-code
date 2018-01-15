cimport numpy as np
from libcpp.vector cimport vector

from ..mesh.pytess cimport PyTess
from ..riemann.riemann cimport RiemannBase
from ..domain.domain_manager cimport DomainManager
from ..containers.containers cimport CarrayContainer
from ..equation_state.equation_state cimport EquationStateBase

ctypedef vector[int] nn           # nearest neighbors
ctypedef vector[nn] nn_vec

#cdef inline bint in_box(double x[3], double r, np.float64_t bounds[2][3], int dim)

cdef class Mesh:

    # initialization parameters
    cdef public int relax_iterations
    cdef public int dim
    cdef public double eta
    cdef public bint regularize
    cdef public int num_neighbors

    cdef public list update_ghost_fields
    cdef bint particle_fields_registered

    cdef public CarrayContainer faces

    cdef PyTess tess
    cdef nn_vec neighbors

    # mesh generation routines
    cpdef reset_mesh(self)
    cpdef tessellate(self, CarrayContainer pc, DomainManager domain_manager)
    cpdef build_geometry(self, CarrayContainer pc, DomainManager domain_manager)
    cpdef relax(self, CarrayContainer particles, DomainManager domain_manager)

    cpdef assign_generator_velocities(self, CarrayContainer particles, EquationStateBase equation_state)
    cpdef assign_face_velocities(self, CarrayContainer particles)
    cpdef update_from_fluxes(self, CarrayContainer particles, RiemannBase riemann, double dt)
