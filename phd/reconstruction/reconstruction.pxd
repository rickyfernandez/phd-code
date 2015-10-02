from containers.containers cimport CarrayContainer, ParticleContainer

cdef class ReconstructionBase:

    cdef _compute(self, ParticleContainer particles, CarrayContainer faces, CarrayContainer left_faces,
            CarrayContainer right_faces, double gamma, double dt)

cdef class PieceWiseConstant(ReconstructionBase):

    cdef _compute(self, ParticleContainer particles, CarrayContainer faces, CarrayContainer left_faces,
            CarrayContainer right_faces, double gamma, double dt)

