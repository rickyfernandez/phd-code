import numpy as np

class RiemannBase(object):
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

    def rotate_state(self, state, theta):

        # only the velocity components are affected
        # under a rotation
        u = state[1,:]
        v = state[2,:]

        # perform the rotation 
        u_tmp =  np.cos(theta)*u + np.sin(theta)*v
        v_tmp = -np.sin(theta)*u + np.cos(theta)*v

        state[1,:] = u_tmp
        state[2,:] = v_tmp

#    def rotate_to_face(self, left_face, right_face, faces_info):
#
#        # The orientation of the face for all faces 
#        theta = faces_info["face angles"]
#
#        # rotate to frame face
#        u_left_rotated =  np.cos(theta)*left_face[1,:] + np.sin(theta)*left_face[2,:]
#        v_left_rotated = -np.sin(theta)*left_face[1,:] + np.cos(theta)*left_face[2,:]
#
#        left_face[1,:] = u_left_rotated
#        left_face[2,:] = v_left_rotated
#
#        u_right_rotated =  np.cos(theta)*right_face[1,:] + np.sin(theta)*right_face[2,:]
#        v_right_rotated = -np.sin(theta)*right_face[1,:] + np.cos(theta)*right_face[2,:]
#
#        right_face[1,:] = u_right_rotated
#        right_face[2,:] = v_right_rotated
#
#    def rotate_to_lab(self, fluxes, faces_info):
#        """
#        """
#
#        # The orientation of the face for all faces 
#        theta = faces_info["face angles"]
#
#        Fu = fluxes[1,:]
#        Fv = fluxes[2,:]
#
#        # rotate back to labrotary frame
#        Fu_lab = np.cos(theta)*Fu - np.sin(theta)*Fv
#        Fv_lab = np.sin(theta)*Fu + np.cos(theta)*Fv
#
#        fluxes[1,:] = Fu_lab
#        fluxes[2,:] = Fv_lab
