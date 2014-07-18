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
        r = np.transpose(particles[indices])
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

        vor = Voronoi(particles)

        num_particles = particles.shape[0]

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


    def faces_for_flux(self, particles, w, particles_index, neighbor_graph, face_graph, voronoi_vertices):

        ngraph = copy.deepcopy(neighbor_graph)
        fgraph = copy.deepcopy(face_graph)

        face_list = []
        for p in particles_index["real"]:

            # skip particles that have all their faces removed
            if not len(ngraph[p]):
                continue

            # grab pointers to voronoi vertices and to neighbors
            # a face is just pointers to voronoi vertices
            faces     = fgraph[p]
            neighbors = ngraph[p]

            # go through neighbors and corresponding faces
            for i, neighbor in enumerate(neighbors):

                # grab voronoi vertices of face corresponding to neighbor
                vor_verts  = voronoi_vertices[np.asarray(faces[i])]

                # creat vector from one voronoi vertex to the other
                # since vertices are ordered counter clockwise
                normal = vor_verts[1] - vor_verts[0]

                # area of face
                area = np.sqrt(normal.dot(normal))

                # rotate by -90 degress to create face normal
                x, y = normal

                # create vector form origin particle to center mass of face
                dr = np.mean(vor_verts, axis=0) - particles[p]
                x1, y1 = dr

                if x1*y - y1*x > 0.0: x, y = y, -x
                else: x, y = -y, x

                # find the angle of the face
                theta = np.angle(x+1j*y)

                # find the velocity of the face
                rl = particles[p]; rr = particles[neighbor]
                wl = w[:,p]; wr = w[:,neighbor]

                f = np.mean(vor_verts, axis=0)  # center mass of face

                w_face = 0.5*(wl + wr)
                w_face += np.sum((wl - wr)*(f-(rr + rl)*0.5))*(rr-rl)/np.sum((rr-rl)**2)

                w_face_x, w_face_y = w_face

                # store the angle area and points left and right of face
                face_list.append([theta, area, w_face_x, w_face_y, p, neighbor])

            # destroy link of neighbors to point 
            for neighbor in neighbors:

                k = ngraph[neighbor].index(p)

                ngraph[neighbor].pop(k)
                fgraph[neighbor].pop(k)

            # destroy link of point to neighbors
            ngraph[p] = []
            fgraph[p] = []


        return np.transpose(np.asarray(face_list))
