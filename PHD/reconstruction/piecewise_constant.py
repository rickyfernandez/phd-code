from reconstruction_base import reconstruction_base
import reconstruct as re
import numpy as np

class piecewise_constant(reconstruction_base):

    def gradient(self):
        return None, None

    def extrapolate(self, left_face, right_face, gradx, grady, faces_info, particles, primitive, w, particles_index, neighbor_graph, neighbor_graph_sizes,
            face_graph, voronoi_vertices):
        pass

