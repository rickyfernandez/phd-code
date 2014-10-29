from hll_2d import Hll2D
import riemann

class Hllc2D(Hll2D):
    """
    2d hllc riemann solver
    """
    def solver(self, left_face, right_face, fluxes, normal, faces_info, gamma, num_faces):
        """
        solve the riemann problem using the 2d hllc solver
        """
        riemann.hllc_2d(left_face, right_face, fluxes, normal, faces_info["velocities"], gamma, num_faces)
