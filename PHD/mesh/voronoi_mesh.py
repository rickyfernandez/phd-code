from scipy.spatial import Voronoi
import cell_volume_center as cv
import numpy as np
import itertools
import copy

class voronoi_mesh(object):

    def regularization(self, prim, particles, gamma, cell_info, particles_index):

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
        Create voronoi tesselation from particle positions
        """

        vor = Voronoi(particles.T)

        num_particles = particles.shape[1]

        # create neighbor and face graph
        face_graph = [[] for i in xrange(num_particles)]
        neighbor_graph = [[] for i in xrange(num_particles)]
        face_graph2 = [[] for i in xrange(num_particles)]

        # loop through each face collecting the two particles
        # that made that face as well as the face itself
        for i, face in enumerate(vor.ridge_points):

            p1, p2 = face
            neighbor_graph[p1].append(p2)
            neighbor_graph[p2].append(p1)

            face_graph[p1].append(vor.ridge_vertices[i])
            face_graph[p2].append(vor.ridge_vertices[i])

            face_graph2[p1] += vor.ridge_vertices[i]
            face_graph2[p2] += vor.ridge_vertices[i]

        # sizes for 1d graphs
        neighbor_graph_sizes = np.array([len(n) for n in neighbor_graph], dtype=np.int32)
        face_graph_sizes = np.array([len(n) for n in face_graph2], dtype=np.int32)

        # graphs in 1d
        neighbor_graph2 = np.array(list(itertools.chain.from_iterable(neighbor_graph)), dtype=np.int32)
        face_graph2 = np.array(list(itertools.chain.from_iterable(face_graph2)), dtype=np.int32)

        return neighbor_graph, face_graph, vor.vertices, neighbor_graph2, neighbor_graph_sizes, face_graph2, face_graph_sizes
#--->   #return neighbor_graph, face_graph, vor.vertices


    def volume_center_mass(self, particles, neighbor_graph, neighbor_graph_size, face_graph, voronoi_vertices,
            particles_index):

        num_particles = particles_index["real"].size
        cell_info = {"volume": np.zeros(num_particles, dtype="float64"), "center of mass": np.zeros((2, num_particles), dtype="float64")}

        cv.cell_volume_center(particles, neighbor_graph, neighbor_graph_size, face_graph, voronoi_vertices,
                cell_info["volume"], cell_info["center of mass"], num_particles)

        return cell_info


    def faces_for_flux(self, particles, w, particles_index, neighbor_graph, neighbor_graph_size, face_graph, voronoi_vertices):

        num_particles = particles_index["real"].size
        num = cv.number_of_faces(neighbor_graph, neighbor_graph_size, num_particles)

        faces_info = np.empty((6, num), dtype="float64")

        cv.faces_for_flux(particles, neighbor_graph, neighbor_graph_size, face_graph, voronoi_vertices, w, faces_info,  num_particles)

        return faces_info
