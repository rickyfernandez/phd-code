from riemann_base import RiemannBase
import numpy as np
import riemann

class Hllc(RiemannBase):

    def solver(self, left_face, right_face, fluxes, w, gamma, num_faces):
        riemann.hllc(left_face, right_face, fluxes, w, gamma, num_faces)
