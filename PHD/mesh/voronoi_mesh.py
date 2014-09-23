from scipy.spatial import Voronoi
import cell_volume_center as cv
import numpy as np
import itertools

class VoronoiMesh2D(object):
    """
    voronoi mesh class
    """

    def assign_particle_velocities(self, particles, primitive, particles_index, cell_info, gamma, regular):
        """
        give particles local fluid velocities, regularization can be added
        """

        # mesh regularization
        if regular == True:
            w = self.regularization(primitive, particles, gamma, cell_info, particles_index)
        else:
            w = np.zeros((2,particles_index["real"].size), dtype="float64")

        # transfer particle velocities to ghost particles
        ghost_map = particles_index["ghost_map"]
        w = np.hstack((w, w[:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))

        # add particle velocities
        w[:, particles_index["real"]]  += primitive[1:3, particles_index["real"]]
        w[:, particles_index["ghost"]] += primitive[1:3, particles_index["ghost"]]

        return w


    def regularization(self, prim, particles, gamma, cell_info, particles_index):
        """
        give particles additional velocity to steer to center of mass
        """

        eta = 0.25

        indices = particles_index["real"]

        pressure = prim[3, indices]
        rho      = prim[0, indices]

        c = np.sqrt(gamma*pressure/rho)

        # generate distance for center mass to particle position
        r = particles[:,indices]
        s = cell_info["center of mass"]

        d = s - r
        d = np.sqrt(np.sum(d**2,axis=0))

        R = np.sqrt(cell_info["volume"]/np.pi)

        w = np.zeros(s.shape)

        i = (0.9 <= d/(eta*R)) & (d/(eta*R) < 1.1)
        if i.any():
            w[:,i] += c[i]*(s[:,i] - r[:,i])*(d[i] - 0.9*eta*R[i])/(d[i]*0.2*eta*R[i])

        j = 1.1 <= d/(eta*R)
        if j.any():
            w[:,j] += c[j]*(s[:,j] - r[:,j])/d[j]

        return w


    def tessellate(self, particles):
        """
        create voronoi tesselation from particle positions
        """

        vor = Voronoi(particles.T)

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


    def volume_center_mass(self, particles, particles_index, graphs):
        """
        find the volume and center of mass for all real particles
        """

        num_particles = particles_index["real"].size

        cell_info = {
                "volume":         np.zeros(num_particles, dtype="float64"),
                "center of mass": np.zeros((2, num_particles), dtype="float64")
                }

        cv.cell_volume_center(particles, graphs["neighbors"], graphs["number of neighbors"], graphs["faces"], graphs["voronoi vertices"],
                cell_info["volume"], cell_info["center of mass"], num_particles)

        return cell_info


    def faces_for_flux(self, particles, primitive, w, particles_index, graphs):
        """
        find the area, orientation, center of mass, and velocity of each face as well
        the particles that share the face and total numbe of faces
        """

        num_real_particles = particles_index["real"].size
        num_faces = cv.number_of_faces(graphs["neighbors"], graphs["number of neighbors"], num_real_particles)

        faces_info = {
                "face angles":         np.empty(num_faces, dtype="float64"),
                "face areas":          np.empty(num_faces, dtype="float64"),
                "face center of mass": np.zeros((2, num_faces), dtype="float64"),
                "face pairs":          np.empty((2, num_faces), dtype="int32"),
                "face velocities":     np.zeros((2, num_faces), dtype="float64"),
                "number faces":        num_faces
                }

        cv.faces_for_flux(faces_info["face areas"], faces_info["face velocities"], faces_info["face angles"], faces_info["face pairs"],
                faces_info["face center of mass"], particles, graphs["neighbors"], graphs["number of neighbors"], graphs["faces"],
                graphs["voronoi vertices"], w, num_real_particles)

        # grab left and right states for each face
        faces_info["left faces"]  = np.ascontiguousarray(primitive[:, faces_info["face pairs"][0,:]])
        faces_info["right faces"] = np.ascontiguousarray(primitive[:, faces_info["face pairs"][1,:]])

        return faces_info

class VoronoiMesh3D(VoronoiMesh2D):

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
