from hll_3d import Hll3D
import numpy as np
import riemann

class Hllc3D(Hll3D):
    """
    3d hllc riemann solver
    """
    def solver(self, left_face, right_face, fluxes, normal, faces_info, gamma, num_faces):
        """
        solve the riemann problem using the 3d hllc solver
        """
        riemann.hllc_3d(left_face, right_face, fluxes, normal, faces_info["velocities"], gamma, num_faces)
