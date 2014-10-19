from hll import Hll
import riemann

class Hllc(Hll):

    """
    Riemann base class. All riemann solvers should inherit this class
    """
    def solver(self, left_face, right_face, fluxes, normal, faces_info, gamma, num_faces):
        riemann.hllc(left_face, right_face, fluxes, normal, faces_info["velocities"], gamma, num_faces)
