from riemann_base import RiemannBase
import numpy as np
import riemann

class Castro(RiemannBase):

    def state(self, left_face, right_face, gamma):

        num_faces = left_face.shape[1]
        state_face = np.zeros((5,num_faces), dtype="float64")
        riemann.castro(left_face, right_face, state_face, gamma, num_faces)

        return state_face
