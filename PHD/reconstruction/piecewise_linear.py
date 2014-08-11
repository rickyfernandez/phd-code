from reconstruct_base import ReconstructBase
import reconstruct as re
import numpy as np

class PiecewiseLinear(ReconstructBase):

    def gradient(self, primitive, particles, particles_index, cell_info, neighbor_graph, neighbor_graph_sizes, face_graph, circum_centers):

        num_real_particles = particles_index["real"].size

        gradx = np.zeros((4, num_real_particles), dtype="float64")
        grady = np.zeros((4, num_real_particles), dtype="float64")

        re.gradient(primitive, gradx, grady, particles, cell_info["volume"], cell_info["center of mass"], neighbor_graph,
                neighbor_graph_sizes, face_graph, circum_centers, num_real_particles)

        return gradx, grady


    def extrapolate(self, left_face, right_face, gradx, grady, faces_info, cell_com, gamma, dt):

        num_faces = left_face.shape[1]

        re.extrapolate(left_face, right_face, gradx, grady, faces_info["face center of mass"], faces_info["face pairs"],
                cell_com, gamma, dt, num_faces)
