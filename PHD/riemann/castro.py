import numpy as np
import riemann
from riemann_base import riemann_base

class castro(riemann_base):

    def state(self, left_face, right_face, gamma):

        num_faces = left_face.shape[1]
        state_face = np.zeros((5,num_faces), dtype="float64")
        riemann.castro(left_face, right_face, state_face, gamma, num_faces)

        return state_face



if __name__ == "__main__":

    left  = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    right = np.array([[0.125, 0.125, 0.125], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])

    print Hllc(left, right, 1.4)
