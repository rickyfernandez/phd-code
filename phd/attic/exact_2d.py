from riemann_base import RiemannBase
import numpy as np
import riemann

class Exact2D(RiemannBase):
    """
    2d exact riemann solver
    """
    def __init__(self, reconstruction=None):
        self.dim = 2
        self.reconstruction = reconstruction


    def get_dt(self, fields, vol, gamma):
        """
        calculate the global timestep for the simulation
        """
        # grab values that correspond to real particles
        dens = fields.get_field("density")
        pres = fields.get_field("pressure")

        # sound speed
        c = np.sqrt(gamma*pres/dens)

        # calculate approx radius of each voronoi cell
        R = np.sqrt(vol/np.pi)

        return np.min(R/c)


    def reconstruct_face_states(self, particles, particles_index, graphs, primitive, cells_info, faces_info, gamma, dt):
        """
        reconstruct primitive particle values to face values
        """
        # construct left and right constant states at each face
        self.left_right_states(primitive, faces_info)

        # pointer to left and right states
        left_face  = faces_info["left faces"]
        right_face = faces_info["right faces"]

        # velocity of all faces
        wx = faces_info["velocities"][0,:]
        wy = faces_info["velocities"][1,:]

        # boost to frame of face
        left_face[1,:] -= wx; right_face[1,:] -= wx
        left_face[2,:] -= wy; right_face[2,:] -= wy

        # calculate gradients for each primitve variable
        self.reconstruction.gradient(primitive, particles, particles_index, cells_info, graphs)

        # reconstruct states at faces - hack for right now
        ghost_map = particles_index["ghost_map"]
        cell_com = np.hstack((cells_info["center of mass"], cells_info["center of mass"][:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))
        self.reconstruction.extrapolate(faces_info, cell_com, gamma, dt)


    def solver(self, left_face, right_face, fluxes, normal, faces_info, gamma, num_faces):
        """
        solve the riemann problem using the 2d hll solver
        """
        riemann.exact_2d(left_face, right_face, fluxes, normal, faces_info["velocities"], gamma, num_faces)
