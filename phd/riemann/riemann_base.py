import numpy as np

class RiemannBase(object):
    """
    Riemann base class. All riemann solvers should inherit this class
    """
    def __init__(self, reconstruction=None):
        self.dim = None
        self.reconstruction = reconstruction


    def left_right_states(self, primitive, faces_info):
        """
        construct left and right primitive values for each face.
        """
        faces_info["left faces"]  = np.ascontiguousarray(primitive[:, faces_info["pairs"][0,:]])
        faces_info["right faces"] = np.ascontiguousarray(primitive[:, faces_info["pairs"][1,:]])


    def fluxes(self, particles, particles_index, graphs, primitive, cells_info, faces_info, gamma, dt):
        """
        reconstruct values at each face then solve the rieman problem returning the fluxes
        """
        # extrapolate primitive variables to com of faces
        self.reconstruct_face_states(particles, particles_index, graphs, primitive, cells_info, faces_info, gamma, dt)

        # allocate storage for fluxes
        num_faces = faces_info["number of faces"]
        fluxes = np.zeros((self.dim+2,num_faces), dtype="float64")

        # solve the riemann problem along face normal then project back lab axes
        self.solver(faces_info["left faces"], faces_info["right faces"], fluxes, faces_info["normal"], faces_info, gamma, num_faces)

        return fluxes


    def get_dt(self, fields, vol, gamma):
        pass


    def reconstruct_face_states(self, particles, particles_index, graphs, primitive, cells_info, faces_info, gamma, dt):
        pass


    def solver(self, left_face, right_face, fluxes, faces_info, gamma, num_faces):
        pass
