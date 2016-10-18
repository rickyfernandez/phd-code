cimport numpy as np

from ..mesh.mesh cimport Mesh
from ..containers.containers cimport CarrayContainer, CarrayContainer

cdef class ReconstructionBase:

    cdef _compute(self, CarrayContainer pc, CarrayContainer faces, CarrayContainer left_faces,
            CarrayContainer right_faces, Mesh mesh, double gamma, double dt)

cdef class PieceWiseConstant(ReconstructionBase):
    pass

cdef class PieceWiseLinear(ReconstructionBase):

    cdef public CarrayContainer grad

    cdef _compute_gradients(self, CarrayContainer pc, CarrayContainer faces, Mesh mesh)
