from scipy.spatial import Voronoi
import numpy as np
import mesh

class VoronoiMeshBase(object):
    """
    voronoi mesh class
    """
    def __init__(self):
        self.dim = None

    def assign_face_velocities(self, particles, particles_index, graphs, faces_info, w):
        """
        give velocities to faces from neighboring particles
        """
        num_real_particles = particles_index["real"].size
        num_faces = faces_info["number of faces"]

        # allocate space for face velocities and the array to faces info dictionary
        faces_info["velocities"] = np.zeros((self.dim, num_faces), dtype="float64")
        self.compute_assign_face_velocities(particles, graphs, faces_info, w, num_real_particles)


    def assign_particle_velocities(self, particles, fields, particles_index, cells_info, gamma, regular):
        """
        give particles local fluid velocities, regularization can be added
        """
        # mesh regularization
        if regular == True:
            w = self.regularization(fields, particles, gamma, cells_info, particles_index)
        else:
            w = np.zeros((self.dim,particles_index["real"].size), dtype="float64")

        # transfer particle velocities to ghost particles - there might be an error here?
        ghost_map = particles_index["ghost_map"]
        w = np.hstack((w, w[:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))

        # add particle velocities
        w[:, particles_index["real"]]  += fields.prim[1:self.dim + 1, particles_index["real"]]
        w[:, particles_index["ghost"]] += fields.prim[1:self.dim + 1, particles_index["ghost"]]

        return w

    def cell_and_faces_info(self, particles, particles_index, graphs):
        """
        compute volume and center of mass of all real particles and compute areas, center of mass, normal
        face pairs, and number of faces for faces
        """
        num_real_particles = particles_index["real"].size
        cells_info = {
                "volume":         np.zeros(num_real_particles, dtype="float64"),
                "center of mass": np.zeros((self.dim, num_real_particles), dtype="float64")
                }

        num_faces = mesh.number_of_faces(graphs["neighbors"], graphs["number of neighbors"], num_real_particles)
        faces_info = {
                "areas":           np.empty(num_faces, dtype="float64"),
                "center of mass":  np.zeros((self.dim, num_faces), dtype="float64"),
                "normal":          np.empty((self.dim, num_faces), dtype="float64"),
                "pairs":           np.empty((2, num_faces), dtype="int32"),
                "number of faces": num_faces
                }

        self.compute_cell_face_info(particles, graphs, cells_info, faces_info, num_real_particles)

        return cells_info, faces_info


    def regularization(self, fields, particles, gamma, cells_info, particles_index):
        """
        give particles additional velocity to steer towards center of mass. This works in 2d and 3d
        """
        eta = 0.25

        indices = particles_index["real"]

        # grab values that correspond to real particles
        dens = fields.get_field("density")
        pres = fields.get_field("pressure")

        # sound speed of all real particles
        c = np.sqrt(gamma*pres/dens)

        # particle positions and center of mass of real particles
        r = particles[:,indices]
        s = cells_info["center of mass"]

        # distance form center mass to particle position
        d = s - r
        d = np.sqrt(np.sum(d**2,axis=0))

        # approximate length of cells
        R = self.cell_length(cells_info["volume"])
        w = np.zeros(s.shape)

        # regularize
        i = (0.9 <= d/(eta*R)) & (d/(eta*R) < 1.1)
        if i.any():
            w[:,i] += c[i]*(s[:,i] - r[:,i])*(d[i] - 0.9*eta*R[i])/(d[i]*0.2*eta*R[i])

        j = 1.1 <= d/(eta*R)
        if j.any():
            w[:,j] += c[j]*(s[:,j] - r[:,j])/d[j]

        return w


    def cell_length(self, vol):
        pass


    def compute_assign_face_velocities(self,):
        pass


    def tessellate(self, particles):
        pass


    def compute_cell_face_info(self, particles, graphs, cells_info, faces_info, num_particles):
        pass
