from scipy.spatial import Voronoi
import cell_volume_center as cv
import numpy as np
import itertools

class VoronoiMeshBase(object):
    """
    voronoi mesh class
    """

    def __init__(self):
        self.dim = None

    def assign_face_velocities(self, particles, particles_index, graphs, faces_info, w):

        num_real_particles = particles_index["real"].size
        num_faces = faces_info["number of faces"]

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

        # transfer particle velocities to ghost particles
        ghost_map = particles_index["ghost_map"]
        w = np.hstack((w, w[:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))

        # add particle velocities
        w[:, particles_index["real"]]  += fields.prim[1:(self.dim + 1), particles_index["real"]]
        w[:, particles_index["ghost"]] += fields.prim[1:(self.dim + 1), particles_index["ghost"]]

        return w

    def cell_and_faces_info(self, particles, particles_index, graphs):

        num_real_particles = particles_index["real"].size

        cells_info = {
                "volume":         np.zeros(num_real_particles, dtype="float64"),
                "center of mass": np.zeros((self.dim, num_real_particles), dtype="float64")
                }

        num_faces = cv.number_of_faces(graphs["neighbors"], graphs["number of neighbors"], num_real_particles)

        faces_info = {
                "angles":          np.empty(num_faces, dtype="float64"),
                "areas":           np.empty(num_faces, dtype="float64"),
                "center of mass":  np.zeros((self.dim, num_faces), dtype="float64"),
                "pairs":           np.empty((self.dim, num_faces), dtype="int32"),
                "number of faces": num_faces
                }

        self.compute_cell_face_info(particles, graphs, cells_info, faces_info, num_real_particles)

        return cells_info, faces_info


    def regularization(self, fields, particles, gamma, cells_info, particles_index):
        """
        give particles additional velocity to steer to center of mass
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
        R = np.sqrt(cells_info["volume"]/np.pi)
        w = np.zeros(s.shape)

        # regularize
        i = (0.9 <= d/(eta*R)) & (d/(eta*R) < 1.1)
        if i.any():
            w[:,i] += c[i]*(s[:,i] - r[:,i])*(d[i] - 0.9*eta*R[i])/(d[i]*0.2*eta*R[i])

        j = 1.1 <= d/(eta*R)
        if j.any():
            w[:,j] += c[j]*(s[:,j] - r[:,j])/d[j]

        return w


    def compute_assign_face_velocities(self,):
        pass


    def tessellate(self, particles):
        """
        create voronoi tesselation from particle positions
        """
        pass


    def compute_cell_face_info(self, particles, graphs, cells_info, faces_info, num_particles):
        pass


class VoronoiMesh2D(VoronoiMeshBase):
    """
    voronoi mesh class
    """

    def __init__(self):
        self.dim = 2


    def compute_assign_face_velocities(self, particles, graphs, faces_info, w, num_real_particles):

        cv.assign_face_velocities(particles, graphs["neighbors"], graphs["number of neighbors"],
                faces_info["center of mass"], faces_info["velocities"], w, num_real_particles)


    def compute_cell_face_info(self, particles, graphs, cells_info, faces_info, num_particles):

        cv.cell_face_info(particles, graphs["neighbors"], graphs["number of neighbors"],
                graphs["faces"], graphs["voronoi vertices"],
                cells_info["volume"], cells_info["center of mass"],
                faces_info["areas"], faces_info["angles"], faces_info["pairs"], faces_info["center of mass"],
                num_particles)


    def tessellate(self, particles):
        """
        create voronoi tesselation from particle positions
        """

        # create the tesselation
        vor = Voronoi(particles.T)

        # total number of particles
        num_particles = particles.shape[1]

        # create neighbor and face graph
        neighbor_graph = [[] for i in xrange(num_particles)]
        face_graph = [[] for i in xrange(num_particles)]

        # loop through each face collecting the two particles
        # that made that face as well as the face itself
        for i, face in enumerate(vor.ridge_points):

            p1, p2 = face
            neighbor_graph[p1].append(p2)
            neighbor_graph[p2].append(p1)

            face_graph[p1] += vor.ridge_vertices[i]
            face_graph[p2] += vor.ridge_vertices[i]

        # sizes for 1d graphs
        neighbor_graph_sizes = np.array([len(n) for n in neighbor_graph], dtype=np.int32)

        # graphs in 1d
        neighbor_graph = np.array(list(itertools.chain.from_iterable(neighbor_graph)), dtype=np.int32)
        face_graph = np.array(list(itertools.chain.from_iterable(face_graph)), dtype=np.int32)

        graphs = {
                "neighbors" : neighbor_graph,
                "number of neighbors" : neighbor_graph_sizes,
                "faces" : face_graph,
                "voronoi vertices" : vor.vertices
                }

        return graphs


class VoronoiMesh3D(VoronoiMesh2D):

    def __init__(self):
        self.dim = 3

    def compute_cell_face_info(self, particles, graphs, cells_info, faces_info, num_particles):

        cv.cell_volume_3d(particles, graphs["neighbors"], graphs["number of neighbors"],
        graphs["faces"], graphs["number of face vertices"], graphs["voronoi vertices"],
        cells_info["volume"], cells_info["center of mass"], num_particles)

    def tessellate(self, particles):
        """
        create voronoi tesselation from particle positions
        """

        # create the voronoi tessellation
        vor = Voronoi(particles.T)

        num_particles = particles.shape[1]

        # list of lists that holds all neighbors of particles
        neighbor_graph = [[] for i in xrange(num_particles)]

        # list of lists that holds all the indices that make up
        # the faces for a given particle
        face_graph = [[] for i in xrange(num_particles)]

        # list of lists that holds then number of vertices for each
        # face for each particle
        face_graph_sizes = [[] for i in xrange(num_particles)]

        # loop through each face collecting the two particles
        # that make up the face as well as the indices that 
        # make up the face
        for i, face in enumerate(vor.ridge_points):

            p1, p2 = face
            neighbor_graph[p1].append(p2)
            neighbor_graph[p2].append(p1)

            # add indices that make up the face
            face_graph[p1] += vor.ridge_vertices[i]
            face_graph[p2] += vor.ridge_vertices[i]

            # add the number of points that make up the face
            face_graph_sizes[p1].append(len(vor.ridge_vertices[i]))
            face_graph_sizes[p2].append(len(vor.ridge_vertices[i]))

        # sizes for 1d graphs, some particles do not have neighbors (coplanar precission error), these
        # are the outside boundaries which does not cause a problem
        #neighbor_graph_sizes = np.array([1 if n == [] else len(n) for n in neighbor_graph], dtype=np.int32)
        neighbor_graph_sizes = np.array([len(n) for n in neighbor_graph], dtype=np.int32)

        # have to clean up the particles that do not have neighbors
        #neighbor_graph = [[-1] if n == [] else n for n in neighbor_graph]

        # there elements with no faces, list must have zero size not empty
        # faces need to be cleaned up too
        #face_graph = [[-1] if x == [] else x for x in face_graph]
        #face_graph_sizes = [[1] if x == [] else x for x in face_graph]

        # graphs in 1d
        neighbor_graph = np.array(list(itertools.chain.from_iterable(neighbor_graph)), dtype=np.int32)
        face_graph = np.array(list(itertools.chain.from_iterable(face_graph)), dtype=np.int32)
        face_graph_sizes = np.array(list(itertools.chain.from_iterable(face_graph_sizes)), dtype=np.int32)

        graphs = {
                "neighbors" : neighbor_graph,
                "number of neighbors" : neighbor_graph_sizes,
                "faces" : face_graph,
                "number of face vertices" : face_graph_sizes,
                "voronoi vertices" : vor.vertices
                }

        return graphs
