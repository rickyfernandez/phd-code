from reconstruct_base import ReconstructBase
import reconstruct as re
import numpy as np

class PiecewiseLinear(ReconstructBase):
    """
    piecewise linear reconstruction with half time step update
    """

    def gradient(self, primitive, particles, particles_index, cells_info, graphs):
        """
        construct gradient for real particles and then assign to ghost particles
        """

        num_real_particles = particles_index["real"].size

        gradx = np.zeros((4, num_real_particles), dtype="float64")
        grady = np.zeros((4, num_real_particles), dtype="float64")

        re.gradient(primitive, gradx, grady, particles, cells_info["volume"], cells_info["center of mass"], graphs["neighbors"],
                graphs["number of neighbors"], graphs["faces"], graphs["voronoi vertices"], num_real_particles)

        self.gradx, self.grady = self.boundary.gradient_to_ghost(particles, gradx, grady, particles_index)


    def extrapolate(self, faces_info, cell_com, gamma, dt):
        """
        linearly predict states to the centroid of face with half time-step
        prediction in time
        """

        num_faces   = faces_info["number of faces"]
        left_faces  = faces_info["left faces"]
        right_faces = faces_info["right faces"]

        re.extrapolate(left_faces, right_faces, self.gradx, self.grady, faces_info["center of mass"], faces_info["pairs"],
                cell_com, gamma, dt, num_faces)
