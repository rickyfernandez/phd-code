import mesh
import itertools
import numpy as np
from scipy.spatial import Voronoi

from utils.particle_tags import ParticleTAGS
from containers.containers import ParticleContainer

class VoronoiMeshBase(object):
    """
    voronoi mesh base class
    """
    def __init__(self, particles):

        self.particles = particles

        self.graph = {
                'neighbors': None,
                'number of neighbors': None,
                'faces': None,
                'voronoi vertices': None,
                }

    def __getitem__(self, name):
        return self.graph[name]

    def tessellate(self):
        pass

    def update_boundary_particles(self):
        pass

class VoronoiMesh2D(VoronoiMeshBase):
    """
    2d voronoi mesh class
    """
    def __init__(self, particles):
        super(VoronoiMesh2D, self).__init__(particles)

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

    def compute_cell_info(self):
        """
        compute volume and center of mass of all real particles and compute areas, center of mass, normal
        face pairs, and number of faces for faces
        """

        num_faces = mesh.number_of_faces(self.particles, self.graph["neighbors"], self.graph["number of neighbors"])
        self.faces.resize(num_faces)

        # the algorithms are cumulative so we have to zero out the data
        self.particles["volume"][:] = 0.0
        self.particles["com-x"][:]  = 0.0
        self.particles["com-y"][:]  = 0.0

        mesh.cell_face_info_2d(self.particles, self.faces, self.graph["neighbors"], self.graph["number of neighbors"],
                self.graph["faces"], self.graph["voronoi vertices"])

    def update_boundary_particles(self):
        cumsum = np.cumsum(self.graph["number of neighbors"], dtype=np.int32)
        mesh.flag_boundary_particles(self.particles, self.graph["neighbors"], self.graph["number of neighbors"], cumsum)

    def update_second_boundary_particles(self):
        cumsum = np.cumsum(self.graph["number of neighbors"], dtype=np.int32)
        mesh.flag_second_boundary_particles(self.particles, self.graph["neighbors"], self.graph["number of neighbors"], cumsum)

    def tessellate(self):
        """
        create 2d voronoi tesselation from particle positions
        """
        pos = np.array([
            self.particles["position-x"],
            self.particles["position-y"]
            ], dtype=np.float64
            )

        # create the tesselation
        vor = Voronoi(pos.T)

        # total number of particles
        num_particles = self.particles.get_number_of_particles()

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

        self.graph["neighbors"] = neighbor_graph
        self.graph["number of neighbors"] = neighbor_graph_sizes
        self.graph["faces"] = face_graph
        self.graph["voronoi vertices"] = vor.vertices

    def build_geometry(self, gamma):

        pc = self.particles

        self.tessellate()
        self.update_boundary_particles()
        self.update_second_boundary_particles() #tmp delete
        self.compute_cell_info()

        indices = np.where(
                (pc['tag']  == ParticleTAGS.Real) |
                (pc['type'] == ParticleTAGS.Boundary) |
                (pc['type'] == ParticleTAGS.BoundarySecond) )[0]

        # now update primitive variables
        vol  = pc['volume'][indices]

        mass = pc['mass'][indices]
        momx = pc['momentum-x'][indices]
        momy = pc['momentum-y'][indices]
        ener = pc['energy'][indices]

        # update primitive variables
        pc['density'][indices]    = mass/vol
        pc['velocity-x'][indices] = momx/mass
        pc['velocity-y'][indices] = momy/mass
        pc['pressure'][indices]   = (ener/vol - 0.5*(mass/vol)*((momx/mass)**2 + (momy/mass)**2))*(gamma-1.0)
