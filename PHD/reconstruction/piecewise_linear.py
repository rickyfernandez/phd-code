from reconstruction_base import reconstruction_base
import reconstruct as re
import numpy as np

class piecewise_linear(reconstruction_base):

    def gradient(self, gradx, grady, primitive, particles, cell_info, neighbor_graph, neighbor_graph_sizes, face_graph, circum_centers):

        num_faces = faces_info.shape[1]
        gradx = np.empty((6, num_faces), dtype="float64")
        grady = np.empty((6, num_faces), dtype="float64")

        num_real_particles = particles_index["real"].size

        re.gradient(primitive, gradx, grady, particles, cell_info["volume"], cell_info["center_of_mass"], neighbor_graph,
                neighbor_graph_sizes, face_graph, circum_centers, num_real_particles)

        return gradx, grady
#-->

    def extrapolate(self, left_face, right_face, grad, faces_info, particles, primitive, w, particles_index, neighbor_graph, neighbor_graph_sizes,
            face_graph, voronoi_vertices):
        pass

