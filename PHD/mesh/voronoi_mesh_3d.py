from scipy.spatial import Voronoi
from voronoi_mesh_base import VoronoiMeshBase
import cell_volume_center as cv
import numpy as np
import itertools


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
