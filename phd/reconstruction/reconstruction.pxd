cimport numpy as np
from ..mesh.mesh cimport Mesh
from ..containers.containers cimport CarrayContainer, ParticleContainer

cdef class ReconstructionBase:

    cdef _compute(self, ParticleContainer pc, CarrayContainer faces, CarrayContainer left_faces,
            CarrayContainer right_faces, Mesh mesh, double gamma, double dt)

cdef class PieceWiseConstant(ReconstructionBase):
    pass

cdef class PieceWiseLinear(ReconstructionBase):

    cdef public CarrayContainer grad

    cdef _compute_gradients(self, ParticleContainer pc, CarrayContainer faces, Mesh mesh)
