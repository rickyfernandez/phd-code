import mesh
import itertools
import numpy as np
from scipy.spatial import Voronoi

from containers.containers import ParticleContainer

class VoronoiMeshBase(dict):
    """
    voronoi mesh class
    """
    def __init__(self, *arg, **kw):
        super(VoronoiMeshBase, self).__init__(*arg, **kw)

        self.dim = None

    def tessellate(self, particles):
        pass

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

        face_vars = {
                "area": "double",
                "velocity-x": "double",
                "velocity-y": "double",
                "normal-x": "double",
                "normal-y": "double",
                "com-x": "double",
                "com-y": "double",
                "pair-i": "longlong",
                "pair-j": "longlong",
                }
        self.faces = ParticleContainer(var_dict=face_vars)

    def compute_cell_info(self, particles):
        """
        compute volume and center of mass of all real particles and compute areas, center of mass, normal
        face pairs, and number of faces for faces
        """

        num_faces = mesh.number_of_faces(particles, self["neighbors"], self["number of neighbors"])
        self.faces.resize(num_faces)

        vol = particles["volume"]; xcom = particles["com-x"]; ycom = particles["com-y"]
        vol[:] = 0.0
        xcom[:] = 0.0
        ycom[:] = 0.0

        mesh.cell_face_info_2d(particles, self.faces, self["neighbors"], self["number of neighbors"],
                self["faces"], self["voronoi vertices"])


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


#class VoronoiMesh3D(VoronoiMeshBase):
#    """
#    3d voronoi mesh class
#    """
#    def __init__(self, *arg, **kw):
#        super(VoronoiMesh3D, self).__init__(*arg, **kw)
#
#        self.dim = 3
#        self["neighbors"] = None
#        self["number of neighbors"] = None
#        self["faces"] = None
#        self["number of face vertices"] = None
#        self["voronoi vertices"] = None
#
##    def cell_length(self, vol):
##        """
##        compute length scale of the cell
##        """
##        return (3.*vol/(4.*np.pi))**(1.0/3.0)
##
##
##    def compute_assign_face_velocities(self, particles, graphs, faces_info, w, num_real_particles):
##        """
##        compute the face velocity from neighboring particles and it's residual motion
##        """
##        mesh.assign_face_velocities_3d(particles, graphs["neighbors"], graphs["number of neighbors"],
##                faces_info["center of mass"], faces_info["velocities"], w, num_real_particles)
##
##
##    def compute_cell_face_info(self, particles, graphs, cells_info, faces_info, num_particles):
##        """
##        compute volume and center of mass of all real particles and compute areas, center of mass, normal
##        face pairs, and number of faces for faces
##        """
##        mesh.cell_face_info_3d(particles, graphs["neighbors"], graphs["number of neighbors"],
##        graphs["faces"], graphs["number of face vertices"], graphs["voronoi vertices"],
##        cells_info["volume"], cells_info["center of mass"],
##        faces_info["areas"], faces_info["normal"], faces_info["pairs"], faces_info["center of mass"],
##        num_particles)
##
##
#    def tessellate(self, particles):
#        """
#        create 3d voronoi tesselation from particle positions
#        """
#        # create the voronoi tessellation
#        vor = Voronoi(particles.T)
#
#        num_particles = particles.shape[1]
#
#        # list of lists that holds all neighbors of particles
#        neighbor_graph = [[] for i in xrange(num_particles)]
#
#        # list of lists that holds all the indices that make up
#        # the faces for a given particle
#        face_graph = [[] for i in xrange(num_particles)]
#
#        # list of lists that holds then number of vertices for each
#        # face for each particle
#        face_graph_sizes = [[] for i in xrange(num_particles)]
#
#        # loop through each face collecting the two particles
#        # that make up the face as well as the indices that 
#        # make up the face
#        for i, face in enumerate(vor.ridge_points):
#
#            p1, p2 = face
#            neighbor_graph[p1].append(p2)
#            neighbor_graph[p2].append(p1)
#
#            # add indices that make up the face
#            face_graph[p1] += vor.ridge_vertices[i]
#            face_graph[p2] += vor.ridge_vertices[i]
#
#            # add the number of points that make up the face
#            face_graph_sizes[p1].append(len(vor.ridge_vertices[i]))
#            face_graph_sizes[p2].append(len(vor.ridge_vertices[i]))
#
#        # sizes for 1d graphs, some particles do not have neighbors (coplanar precission error), these
#        # are the outside boundaries which does not cause a problem
#        neighbor_graph_sizes = np.array([len(n) for n in neighbor_graph], dtype=np.int32)
#
#        # graphs in 1d
#        neighbor_graph = np.array(list(itertools.chain.from_iterable(neighbor_graph)), dtype=np.int32)
#        face_graph = np.array(list(itertools.chain.from_iterable(face_graph)), dtype=np.int32)
#        face_graph_sizes = np.array(list(itertools.chain.from_iterable(face_graph_sizes)), dtype=np.int32)
#
#        self["neighbors"] = neighbor_graph
#        self["number of neighbors"] = neighbor_graph_sizes
#        self["faces"] = face_graph
#        self["number of face vertices"] = face_graph_sizes
#        self["voronoi vertices"] = vor.vertices
