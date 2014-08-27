import numpy as np

class RiemannBase(object):
    """
    Riemann base class. All riemann solvers should inherit this class
    """
    def __init__(self, reconstruction=None):
        self.reconstruction = reconstruction

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

    def fluxes(self, faces_info, gamma, dt, cell_info, particles_index):

        left_face  = faces_info["left faces"]
        right_face = faces_info["right faces"]

        #num_faces = left_face.shape[1]
        num_faces = faces_info["number faces"]
        fluxes = np.zeros((4,num_faces), dtype="float64")

        # The orientation of the face for all faces 
        theta = faces_info["face angles"]

        # velocity of all faces
        wx = faces_info["face velocities"][0,:]
        wy = faces_info["face velocities"][1,:]

        # velocity of the faces in the direction of the faces
        # dot product invariant under rotation
        w = wx*np.cos(theta) + wy*np.sin(theta)

        # reconstruct to states to faces
        # hack for right now
        ghost_map = particles_index["ghost_map"]
        cell_com = np.hstack((cell_info["center of mass"], cell_info["center of mass"][:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))
        self.reconstruction.extrapolate(left_face, right_face, faces_info, cell_com, gamma, dt)

        # rotate to face frame 
        self.rotate_state(left_face,  theta)
        self.rotate_state(right_face, theta)

        # solve the riemann problem
        self.solver(left_face, right_face, fluxes, w, gamma, num_faces)

        # rotate the flux back to the lab frame 
        self.rotate_state(fluxes, -theta)

        return fluxes

    def solver(self, left_face, right_face, fluxes, w, gamma, num_faces):
        pass
