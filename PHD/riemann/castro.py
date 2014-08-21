from riemann_base import RiemannBase
import numpy as np
import riemann

class Castro(RiemannBase):

    def state(self, left_face, right_face, gamma):

        num_faces = left_face.shape[1]
        state_face = np.zeros((5,num_faces), dtype="float64")
        riemann.castro(left_face, right_face, state_face, gamma, num_faces)

        return state_face

    def fluxes(self, left_face, right_face, faces_info, gamma):

        num_faces = left_face.shape[1]
        fluxes = np.zeros((4,num_faces), dtype="float64")

        # The orientation of the face for all faces 
        theta = faces_info["face angles"]

        # velocity of all faces
        wx = faces_info["face velocities"][0,:]
        wy = faces_info["face velocities"][1,:]

        # velocity of the faces in the direction of the faces
        # dot product invariant under rotation
        w = wx*np.cos(theta) + wy*np.sin(theta)

        # rotate to face frame 
        # rotate states to the frame of the face
        self.rotate_state(left_face,  theta)
        self.rotate_state(right_face, theta)

        # solve the riemann problem
        riemann.hllc(left_face, right_face, fluxes, w, gamma, num_faces)

        # rotate the flux back to the lab frame 
        self.rotate_state(fluxes, -theta)

        return fluxes
