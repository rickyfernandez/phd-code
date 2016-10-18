from ..utils.particle_tags import ParticleTAGS

import itertools
import numpy as np
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon
import matplotlib.colors as mat_colors
from matplotlib.collections import PatchCollection

def vor_collection(pc, field):

    mesh = VoronoiMesh2D()
    mesh.tessellate(pc['position-x'], pc['position-y'])

    patch = []; colors = []

    x = pc['position-x']
    y = pc['position-y']
    tags = pc['tag']

    ii = 0; jj = 0
    for i in range(x.size):

        jj += mesh['number of neighbors'][i]*2

        if tags[i] == ParticleTAGS.Real:

            verts_indices = np.unique(mesh['faces'][ii:jj])
            verts = mesh['voronoi vertices'][verts_indices]

            # coordinates of neighbors relative to particle p
            xc = verts[:,0] - x[i]
            yc = verts[:,1] - y[i]

            # sort in counter clock wise order
            sorted_vertices = np.argsort(np.angle(xc+1j*yc))
            verts = verts[sorted_vertices]

            patch.append(Polygon(verts, True))

        ii = jj

    for i in range(x.size):
        if tags[i] == ParticleTAGS.Real:
            colors.append(pc[field][i])

    return patch, colors

class VoronoiMesh2D(dict):
    """
    2d voronoi mesh class
    """
    def __init__(self):

        self["neighbors"] = None
        self["number of neighbors"] = None
        self["faces"] = None
        self["voronoi vertices"] = None

    def tessellate(self, x, y):
        """
        create 2d voronoi tesselation from particle positions
        """
        particles = np.array([x, y])

        # create the tesselation
        vor = Voronoi(particles.T)

        # total number of particles
        num_particles = x.size

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

