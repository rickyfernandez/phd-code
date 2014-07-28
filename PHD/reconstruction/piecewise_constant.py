from reconstruction_base import reconstruction_base
import reconstruct as re
import numpy as np

class piecewise_constant(reconstruction_base):

    def gradient(self):
        return None

    def extrapolate(self, particles, primitive, grad, w, particles_index, neighbor_graph, neighbor_graph_sizes,
            face_graph, voronoi_vertices):

        faces_info = self.faces_for_flux(particles, w, particles_index, neighbor_graph, neighbor_graph_sizes, face_graph, voronoi_vertices)

        # grab left and right states
        left  = primitive[:, faces_info[4,:].astype(int)]
        right = primitive[:, faces_info[5,:].astype(int)]

        return left, right, faces_info

    def faces_for_flux(self, particles, w, particles_index, neighbor_graph, neighbor_graph_size, face_graph, voronoi_vertices):

        num_particles = particles_index["real"].size
        num = re.number_of_faces(neighbor_graph, neighbor_graph_size, num_particles)

        faces_info = np.empty((6, num), dtype="float64")

        re.faces_for_flux(particles, neighbor_graph, neighbor_graph_size, face_graph, voronoi_vertices, w, faces_info,  num_particles)

        return faces_info
