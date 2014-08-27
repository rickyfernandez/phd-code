from riemann_base import RiemannBase
import numpy as np
import riemann

class Hll(RiemannBase):

    def solver(self, left_face, right_face, fluxes, w, gamma, num_faces):
        riemann.hll(left_face, right_face, fluxes, w, gamma, num_faces)
