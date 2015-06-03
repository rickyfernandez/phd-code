from voronoi_mesh_base import VoronoiMeshBase
from scipy.spatial import Voronoi
import numpy as np
import itertools
#import mesh


class VoronoiMesh2D(VoronoiMeshBase):
    """
    2d voronoi mesh class
    """
    def __init__(self, *arg, **kw):
        super(VoronoiMesh2D, self).__init__(*arg, **kw)

        self.dim = 2
        self["neighbors"] = None
        self["number of neighbors"] = None
        self["faces"] = None
        self["voronoi vertices"] = None


#    def cell_length(self, vol):
#        """
#        compute length scale of the cell
#        """
#        return np.sqrt(vol/np.pi)
#
#
#    def compute_assign_face_velocities(self, particles, graphs, faces_info, w, num_real_particles):
#        """
#        compute the face velocity from neighboring particles and it's residual motion
#        """
#        mesh.assign_face_velocities_2d(particles, graphs["neighbors"], graphs["number of neighbors"],
#                faces_info["center of mass"], faces_info["velocities"], w, num_real_particles)
#
#
#    def compute_cell_face_info(self, particles, graphs, cells_info, faces_info, num_particles):
#        """
#        compute volume and center of mass of all real particles and compute areas, center of mass, normal
#        face pairs, and number of faces for faces
#        """
#        mesh.cell_face_info_2d(particles, graphs["neighbors"], graphs["number of neighbors"],
#                graphs["faces"], graphs["voronoi vertices"],
#                cells_info["volume"], cells_info["center of mass"],
#                faces_info["areas"], faces_info["normal"], faces_info["pairs"], faces_info["center of mass"],
#                num_particles)
#
#
    def tessellate(self, particles):
        """
        create 2d voronoi tesselation from particle positions
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

        self["neighbors"] = neighbor_graph
        self["number of neighbors"] = neighbor_graph_sizes
        self["faces"] = face_graph
        self["voronoi vertices"] = vor.vertices
