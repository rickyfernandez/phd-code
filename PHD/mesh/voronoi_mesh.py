from scipy.spatial import Voronoi
import cell_volume_center as cv
import numpy as np
import itertools
import copy

class voronoi_mesh(object):

    def __init__(self, regularization):
        self.regularization = regularization

    def assign_particle_velocities(self, particles, primitive, particles_index, cell_info, gamma):

        # mesh regularization
        if self.regularization == True:
            w = self.mesh.regularization(primitive, particles, gamma, cell_info, particles_index)
        else:
            w = np.zeros((2,particles_index["real"].size))

        # transfer particle velocities to ghost particles
        ghost_map = particles_index["ghost_map"]
        w = np.hstack((w, w[:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))

        # add particle velocities
        w[:, particles_index["real"]]  += primitive[1:3, particles_index["real"]]
        w[:, particles_index["ghost"]] += primitive[1:3, particles_index["ghost"]]

        return w


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
        #face_graph = [[] for i in xrange(num_particles)]
        neighbor_graph = [[] for i in xrange(num_particles)]
        face_graph2 = [[] for i in xrange(num_particles)]

        # loop through each face collecting the two particles
        # that made that face as well as the face itself
        for i, face in enumerate(vor.ridge_points):

            p1, p2 = face
            neighbor_graph[p1].append(p2)
            neighbor_graph[p2].append(p1)

            #face_graph[p1].append(vor.ridge_vertices[i])
            #face_graph[p2].append(vor.ridge_vertices[i])

            face_graph2[p1] += vor.ridge_vertices[i]
            face_graph2[p2] += vor.ridge_vertices[i]

        # sizes for 1d graphs
        neighbor_graph_sizes = np.array([len(n) for n in neighbor_graph], dtype=np.int32)
        face_graph_sizes = np.array([len(n) for n in face_graph2], dtype=np.int32)

        # graphs in 1d
        neighbor_graph2 = np.array(list(itertools.chain.from_iterable(neighbor_graph)), dtype=np.int32)
        face_graph2 = np.array(list(itertools.chain.from_iterable(face_graph2)), dtype=np.int32)

        #return neighbor_graph, face_graph, vor.vertices, neighbor_graph2, neighbor_graph_sizes, face_graph2, face_graph_sizes
        return neighbor_graph2, neighbor_graph_sizes, face_graph2, face_graph_sizes, vor.vertices
#--->   #return neighbor_graph, face_graph, vor.vertices


    def volume_center_mass(self, particles, neighbor_graph, neighbor_graph_size, face_graph, voronoi_vertices,
            particles_index):

        num_particles = particles_index["real"].size
        cell_info = {"volume": np.zeros(num_particles, dtype="float64"), "center of mass": np.zeros((2, num_particles), dtype="float64")}

        cv.cell_volume_center(particles, neighbor_graph, neighbor_graph_size, face_graph, voronoi_vertices,
                cell_info["volume"], cell_info["center of mass"], num_particles)

        return cell_info


    def faces_for_flux(self, particles, primitive, w, particles_index, neighbor_graph, neighbor_graph_size, face_graph, voronoi_vertices):

        num_real_particles = particles_index["real"].size
        num_faces = cv.number_of_faces(neighbor_graph, neighbor_graph_size, num_real_particles)

        faces_info = {
                "face angles":         np.empty(num_faces, dtype="float64"),
                "face areas":          np.empty(num_faces, dtype="float64"),
                "face center of mass": np.empty((2, num_faces), dtype="float64"),
                "face pairs":          np.empty((2, num_faces), dtype="int32"),
                "face velocities":     np.empty((2, num_faces), dtype="float64")
                }

        cv.faces_for_flux(faces_info["face areas"], faces_info["face velocities"], faces_info["face angles"], faces_info["face pairs"],
                faces_info["face center of mass"], particles, neighbor_graph, neighbor_graph_size, face_graph, voronoi_vertices,
                w, num_real_particles)

        # grab left and right states
        #left  = primitive[:, faces_info[4,:].astype(int)]
        #right = primitive[:, faces_info[5,:].astype(int)]
        left  = primitive[:, faces_info["face pairs"][0,:]]
        right = primitive[:, faces_info["face pairs"][1,:]]

        return left, right, faces_info

    def transform_to_face(self, left_face, right_face, faces_info):
        """
        Transform coordinate system to the state of each face. This
        has two parts. First a boost than a rotation.
        """

        # velocity of all faces
        #wx = faces_info[2,:]; wy = faces_info[3,:]
        wx = faces_info["face velocities"][0,:]
        wy = faces_info["face velocities"][1,:]

        # boost to frame of face
        left_face[1,:] -= wx; right_face[1,:] -= wx
        left_face[2,:] -= wy; right_face[2,:] -= wy


    def rotate_to_face(self, left_face, right_face, faces_info):

        # The orientation of the face for all faces 
        #theta = faces_info[0,:]
        theta = faces_info["face angles"]

        # rotate to frame face
        u_left_rotated =  np.cos(theta)*left_face[1,:] + np.sin(theta)*left_face[2,:]
        v_left_rotated = -np.sin(theta)*left_face[1,:] + np.cos(theta)*left_face[2,:]

        left_face[1,:] = u_left_rotated
        left_face[2,:] = v_left_rotated

        u_right_rotated =  np.cos(theta)*right_face[1,:] + np.sin(theta)*right_face[2,:]
        v_right_rotated = -np.sin(theta)*right_face[1,:] + np.cos(theta)*right_face[2,:]

        right_face[1,:] = u_right_rotated
        right_face[2,:] = v_right_rotated


    def transform_to_lab(self, face_states, faces_info):
        """
        Calculate the flux in the lab frame using the state vector of the face.
        """

        rho = face_states[0,:]
        u   = face_states[1,:]
        v   = face_states[2,:]
        rhoe= face_states[3,:]
        p   = face_states[4,:]

        # The orientation of the face for all faces 
        #theta = faces_info[0,:]
        theta = faces_info["face angles"]

        # velocity of all faces
        #wx = faces_info[2,:]; wy = faces_info[3,:]
        wx = faces_info["face velocities"][0,:]
        wy = faces_info["face velocities"][1,:]

        # components of the flux vector
        F = np.zeros((4, rho.size))
        G = np.zeros((4, rho.size))

        # rotate state back to labrotary frame
        u_lab = np.cos(theta)*u - np.sin(theta)*v
        v_lab = np.sin(theta)*u + np.cos(theta)*v

        # unboost
        u_lab += wx
        v_lab += wy

        # calculate energy density in lab frame
        E = 0.5*rho*(u_lab**2 + v_lab**2) + rhoe

        # flux component in the x-direction
        F[0,:] = rho*(u_lab - wx)
        F[1,:] = rho*u_lab*(u_lab-wx) + p
        F[2,:] = rho*v_lab*(u_lab-wx)
        F[3,:] = E*(u_lab-wx) + p*u_lab

        # flux component in the y-direction
        G[0,:] = rho*(v_lab - wy)
        G[1,:] = rho*u_lab*(v_lab-wy)
        G[2,:] = rho*v_lab*(v_lab-wy) + p
        G[3,:] = E*(v_lab-wy) + p*v_lab

        # dot product flux in orientation of face
        return np.cos(theta)*F + np.sin(theta)*G
