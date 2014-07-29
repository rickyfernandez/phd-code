from reconstruction_base import reconstruction_base
import reconstruct as re
import numpy as np

class piecewise_constant(reconstruction_base):
    pass

#    def gradient(self):
#        return None

#    def extrapolate(self, particles, primitive, grad, w, particles_index, neighbor_graph, neighbor_graph_sizes,
#            face_graph, voronoi_vertices):
#
#        faces_info = self.faces_for_flux(particles, w, particles_index, neighbor_graph, neighbor_graph_sizes, face_graph, voronoi_vertices)
#
#        # grab left and right states
#        left  = primitive[:, faces_info[4,:].astype(int)]
#        right = primitive[:, faces_info[5,:].astype(int)]
#
#        return left, right, faces_info
