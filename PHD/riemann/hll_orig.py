from hll import Hll
import riemann

class Hllc(Hll):

    def solver(self, left_face, right_face, fluxes, w, gamma, num_faces):
        riemann.hllc(left_face, right_face, fluxes, w, gamma, num_faces)
