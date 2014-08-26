from reconstruct_base import ReconstructBase
import reconstruct as re
import numpy as np

class PiecewiseConstant(ReconstructBase):

    def gradient(self, primitive, particles, particles_index, cell_info, neighbor_graph, neighbor_graph_sizes, face_graph, circum_centers):
        pass

    def extrapolate(self, left_face, right_face, faces_info, cell_com, gamma, dt):
        pass

