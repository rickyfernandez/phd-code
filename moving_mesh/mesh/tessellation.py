from scipy.spatial import Voronoi
import numpy as np

def tessellation(particles):

    vor = Voronoi(particles)

    num_particles = particles.shape[0]

    # create neighbor and face graph
    face_graph = [[] for i in xrange(num_particles)]
    neighbor_graph = [[] for i in xrange(num_particles)]

    # loop through each face collecting the two particles
    # that made that face as well as the face itself
    for i, face in enumerate(vor.ridge_points):

        p1, p2 = face
        neighbor_graph[p1].append(p2)
        neighbor_graph[p2].append(p1)

        face_graph[p1].append(vor.ridge_vertices[i])
        face_graph[p2].append(vor.ridge_vertices[i])

    return neighbor_graph, face_graph, vor.vertices
