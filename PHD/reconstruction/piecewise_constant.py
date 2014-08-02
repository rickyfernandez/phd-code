from reconstruction_base import reconstruction_base
import reconstruct as re
import numpy as np

class piecewise_constant(reconstruction_base):

    def gradient(self, primitive, particles, particles_index, cell_info, neighbor_graph, neighbor_graph_sizes, face_graph, circum_centers):
        return None, None

    #def extrapolate(self, left_face, right_face, gradx, grady, faces_info, particles, primitive, w, particles_index, neighbor_graph, neighbor_graph_sizes,
    #        face_graph, voronoi_vertices):
    #def extrapolate(self, left_face, right_face, gradx, grady, faces_info, particles):
    def extrapolate(self, left_face, right_face, gradx, grady, faces_info, cell_com, gamma, dt):
        pass

