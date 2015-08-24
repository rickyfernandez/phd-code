from reconstruct_base import ReconstructBase
import reconstruction as re
import numpy as np

class PiecewiseLinear3D(ReconstructBase):
    """
    3d piecewise linear reconstruction with half time step update
    """
    def gradient(self, primitive, particles, particles_index, cells_info, graphs):
        """
        construct gradient for real particles and then assign to ghost particles
        """
        num_real_particles = particles_index["real"].size

        grad = {
                "x": np.zeros((5, num_real_particles), dtype="float64"),
                "y": np.zeros((5, num_real_particles), dtype="float64"),
                "z": np.zeros((5, num_real_particles), dtype="float64")
                }

        re.gradient_3d(primitive, grad["x"], grad["y"], grad["z"], particles, cells_info["volume"], cells_info["center of mass"], graphs["neighbors"],
                graphs["number of neighbors"], graphs["faces"], graphs["number of face vertices"], graphs["voronoi vertices"], num_real_particles)

        self.grad = self.boundary.gradient_to_ghost(particles, grad, particles_index)


    def extrapolate(self, faces_info, cell_com, gamma, dt):
        """
        linearly predict states to the centroid of face with half time-step
        prediction in time
        """
        num_faces   = faces_info["number of faces"]
        left_faces  = faces_info["left faces"]
        right_faces = faces_info["right faces"]

        re.extrapolate_3d(left_faces, right_faces, self.grad["x"], self.grad["y"], self.grad["z"], faces_info["center of mass"], faces_info["pairs"],
                cell_com, gamma, dt, num_faces)
