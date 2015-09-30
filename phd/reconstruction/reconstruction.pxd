from containers.containers cimport CarrayContainer, ParticleContainer

cdef class ReconstructionBase:

    cdef compute(self, ParticleContainer particles, CarrayContainer faces, CarrayContainer left_faces,
            CarrayContainer right_faces, double gamma, double dt)

cdef class PieceWiseConstant(ReconstructionBase):
    pass

