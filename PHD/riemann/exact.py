from riemann_base import RiemannBase
import numpy as np
import riemann

class Exact(RiemannBase):

    def get_dt(self, fields, vol, gamma):

        # grab values that correspond to real particles
        dens = fields.get_field("density")
        pres = fields.get_field("pressure")

        # sound speed
        c = np.sqrt(gamma*pres/dens)

        # calculate approx radius of each voronoi cell
        R = np.sqrt(vol/np.pi)

        return np.min(R/c)


    def fluxes(self, primitive, faces_info, gamma, dt, cell_info, particles_index):

        self.left_right_states(primitive, faces_info)

        left_face  = faces_info["left faces"]
        right_face = faces_info["right faces"]

        num_faces = faces_info["number faces"]
        face_states = np.zeros((4,num_faces), dtype="float64")

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

        d = face_states[0,:]
        u = face_states[1,:]
        v = face_states[2,:]
        p = face_states[3,:]

        # rotate state back to labrotary frame
        u_lab = np.cos(theta)*u - np.sin(theta)*v
        v_lab = np.sin(theta)*u + np.cos(theta)*v

        # unboost
        u_lab += wx
        v_lab += wy

        # calculate energy density in lab frame
        E = 0.5*d*(u_lab**2 + v_lab**2) + p/(gamma - 1)

        # components of the flux vector
        F = np.zeros((4, d.size))
        G = np.zeros((4, d.size))

        # flux component in the x-direction
        F[0,:] = d*(u_lab - wx)
        F[1,:] = d*u_lab*(u_lab - wx) + p
        F[2,:] = d*v_lab*(u_lab - wx)
        F[3,:] = E*(u_lab - wx) + p*u_lab

        # flux component in the y-direction
        G[0,:] = d*(v_lab - wy)
        G[1,:] = d*u_lab*(v_lab - wy)
        G[2,:] = d*v_lab*(v_lab - wy) + p
        G[3,:] = E*(v_lab - wy) + p*v_lab

        # dot product flux in orientation of face
        return np.cos(theta)*F + np.sin(theta)*G
