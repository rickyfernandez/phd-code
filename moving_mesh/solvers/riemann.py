import numpy as np

class riemann(object):
    """
    Riemann base class. All riemann solvers should inherit this class
    """

    def __init__(self, smallp=1.0E-10, smallc=1.0E-10, smallrho=1.0E-10):
        """
        Set parameters for riemann class.
        """

        self.small_pressure = smallp
        self.small_sound_speed = smallc
        self.small_rho = smallrho


    def _transform_to_face(self, ql, qr, faces_info):
        """
        Transform coordinate system to the state of each face. This
        has two parts. First a boost than a rotation.
        """

        # The orientation of the face for all faces 
        theta = faces_info[0,:]

        # velocity of all faces
        wx = faces_info[2,:]; wy = faces_info[3,:]

        # prepare faces to rotate and boost to face frame
        left_face = ql.copy(); right_face = qr.copy()

        # boost to frame of face
        left_face[1,:] -= wx; right_face[1,:] -= wx
        left_face[2,:] -= wy; right_face[2,:] -= wy

        # rotate to frame face
        u_left_rotated =  np.cos(theta)*left_face[1,:] + np.sin(theta)*left_face[2,:]
        v_left_rotated = -np.sin(theta)*left_face[1,:] + np.cos(theta)*left_face[2,:]

        left_face[1,:] = u_left_rotated
        left_face[2,:] = v_left_rotated

        u_right_rotated =  np.cos(theta)*right_face[1,:] + np.sin(theta)*right_face[2,:]
        v_right_rotated = -np.sin(theta)*right_face[1,:] + np.cos(theta)*right_face[2,:]

        right_face[1,:] = u_right_rotated
        right_face[2,:] = v_right_rotated

        return left_face, right_face

    def _transform_flux_to_lab(self, rho, u, v, rhoe, p, faces_info):
        """
        Calculate the flux in the lab frame using the state vector of the face. 
        """

        # The orientation of the face for all faces 
        theta = faces_info[0,:]

        # velocity of all faces
        wx = faces_info[2,:]; wy = faces_info[3,:]

        # components of the flux vector
        F = np.zeros((4, rho.size))
        G = np.zeros((4, rho.size))

        # rotate state back to labrotary frame
        u = np.cos(theta)*u - np.sin(theta)*v
        v = np.sin(theta)*u + np.cos(theta)*v

        # unboost
        u = u + wx
        v = v + wy

        # calculate energy density in lab frame
        E = 0.5*rho*(u**2 + v**2) + rhoe

        # flux component in the x-direction
        F[0,:] = rho*(u - wx)
        F[1,:] = rho*u*(u-wx) + p
        F[2,:] = rho*v*(u-wx)
        F[3,:] = E*(u-wx) + p*u

        # flux component in the y-direction
        G[0,:] = rho*(v - wy)
        G[1,:] = rho*u*(v-wy)
        G[2,:] = rho*v*(v-wy) + p
        G[3,:] = E*(v-wy) + p*v

        # dot product flux in orientation of face
        return np.cos(theta)*F + np.sin(theta)*G
