from particles.particle_array cimport ParticleArray

cdef class ReconstructionBase:

    cdef compute(self, ParticleArray pa, ParticleArray faces, ParticleArray left_faces,
            ParticleArray right_faces, double gamma, double dt)

cdef class PieceWiseConstant(ReconstructionBase):
    pass

