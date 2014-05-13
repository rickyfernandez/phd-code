from scipy.spatial import Voronoi
import numpy as np
import copy

class voronoi_mesh(object):

    def tessellate(self, particles):
        """
        Create voronoi tesselation from particle positions
        """

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
    

    def _cell_volume_center(self, particle_id, particles, neighbor_graph, face_graph, circum_centers):
        """
        Calculate the volume and center mass of particle.
        """

        # array of indices pairs of voronoi vertices
        # for each face - (numfaces, 2)
        voronoi_faces = np.array(face_graph[particle_id])

        area = circum_centers[voronoi_faces]
        area = area[:,0,:] - area[:,1,:]
        area = (area*area).sum(axis=1)
        np.sqrt(area, area)

        f = np.mean(circum_centers[voronoi_faces], axis=1)
        center_of_mass = 2.0*f/3.0 + particles[particle_id]/3.0

        # coordinates of voronoi generating point
        center    = particles[particle_id]
        neighbors = particles[neighbor_graph[particle_id]]

        # speration vectors form neighbors to voronoi generating point
        r = center - neighbors
        h = 0.5*np.sqrt(np.sum(r**2, axis=1))

        volumes     = 0.5*area*h
        cell_volume = np.sum(0.5*area*h)

        # cell center of mass coordinates
        cm = (center_of_mass*volumes[:,np.newaxis]).sum(axis=0)/cell_volume

        return cell_volume, cm[0], cm[1]

    def volume_center_mass(self, particles, neighbor_graph, particles_index, face_graph, voronoi_vertices):
        """
        Caculate the volumes and center mass of all particles inside the domain.
        """

        # calculate volume of real particles 
        vals = np.empty((3, particles_index["real"].shape[0]), dtype="float64")
        for i, particle_id in enumerate(particles_index["real"]):
            vals[:,i] = self._cell_volume_center(particle_id, particles, neighbor_graph, face_graph, voronoi_vertices)
        return vals


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
