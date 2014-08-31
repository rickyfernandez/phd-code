from riemann_base import RiemannBase
import numpy as np
import riemann

class Exact(RiemannBase):

    def fluxes(self, faces_info, gamma, dt, cell_info, particles_index):

        left_face  = faces_info["left faces"]
        right_face = faces_info["right faces"]

        num_faces = faces_info["number faces"]
        face_states = np.zeros((5,num_faces), dtype="float64")

        # The orientation of the face for all faces 
        theta = faces_info["face angles"]

        # velocity of all faces
        wx = faces_info["face velocities"][0,:]
        wy = faces_info["face velocities"][1,:]

        # boost to frame of face
        left_face[1,:] -= wx; right_face[1,:] -= wx
        left_face[2,:] -= wy; right_face[2,:] -= wy

        # reconstruct to states to faces
        # hack for right now
        ghost_map = particles_index["ghost_map"]
        cell_com = np.hstack((cell_info["center of mass"], cell_info["center of mass"][:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))
        self.reconstruction.extrapolate(faces_info, cell_com, gamma, dt)

        # rotate to face frame 
        self.rotate_state(left_face,  theta)
        self.rotate_state(right_face, theta)

        # solve the riemann problem
        riemann.exact(left_face, right_face, face_states, gamma, num_faces)

        rho = face_states[0,:]
        u   = face_states[1,:]
        v   = face_states[2,:]
        rhoe= face_states[3,:]
        p   = face_states[4,:]

        # rotate state back to labrotary frame
        u_lab = np.cos(theta)*u - np.sin(theta)*v
        v_lab = np.sin(theta)*u + np.cos(theta)*v

        # unboost
        u_lab += wx
        v_lab += wy

        # calculate energy density in lab frame
        E = 0.5*rho*(u_lab**2 + v_lab**2) + rhoe

        # components of the flux vector
        F = np.zeros((4, rho.size))
        G = np.zeros((4, rho.size))

        # flux component in the x-direction
        F[0,:] = rho*(u_lab - wx)
        F[1,:] = rho*u_lab*(u_lab-wx) + p
        F[2,:] = rho*v_lab*(u_lab-wx)
        F[3,:] = E*(u_lab-wx) + p*u_lab

        # flux component in the y-direction
        G[0,:] = rho*(v_lab - wy)
        G[1,:] = rho*u_lab*(v_lab-wy)
        G[2,:] = rho*v_lab*(v_lab-wy) + p
        G[3,:] = E*(v_lab-wy) + p*v_lab

        # dot product flux in orientation of face
        return np.cos(theta)*F + np.sin(theta)*G
