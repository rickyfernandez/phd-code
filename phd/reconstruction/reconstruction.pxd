cimport numpy as np
from mesh.mesh cimport MeshBase
from containers.containers cimport CarrayContainer, ParticleContainer

cdef class ReconstructionBase:

    cdef _compute(self, ParticleContainer particles, CarrayContainer faces, CarrayContainer left_faces,
            CarrayContainer right_faces, MeshBase mesh, double gamma, double dt)

cdef class PieceWiseConstant(ReconstructionBase):
    pass


# out of comission - adding cgal library
#cdef class PieceWiseLinear(ReconstructionBase):
#
#    cdef public dict state_vars
#    cdef public CarrayContainer gradx
#    cdef public CarrayContainer grady
#
#    cdef _compute(self, ParticleContainer particles, CarrayContainer faces, CarrayContainer left_faces,
#            CarrayContainer right_faces, Mesh2d mesh, double gamma, double dt)
#
#    cdef _compute_gradients(self, ParticleContainer particles, CarrayContainer faces, np.int32_t[:] neighbor_graph, np.int32_t[:] num_neighbors,
#        np.int32_t[:] face_graph, double[:,::1] circum_centers)
