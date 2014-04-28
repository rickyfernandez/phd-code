import numpy as np
import copy

def faces_for_flux(particles, w, particles_index, neighbor_graph, face_graph, voronoi_vertices):

    #ngraph = copy.deepcopy(neighbor_graph)
    #fgraph = copy.deepcopy(face_graph)

    # new face graph except faces are not repeated by using logical indices
    #faces_mask = [~np.in1d(neighbor_graph[i], xrange(i)) for i in xrange(len(neighbor_graph))]
    #faces_mask = [~np.in1d(neighbor_graph[i], xrange(i)) for i in particles_index["real"]]


    faces_mask = [np.asarray(neighbor_graph[i]) > i for i in particles_index["real"]]
    num_faces = sum([np.sum(faces_mask[i]) for i in xrange(len(faces_mask))])

    faces_info = np.empty((6,num_faces), dtype=np.float64)

    #face_list = []
    k_new = k_old = 0
    for p in particles_index["real"]:

        # skip particles that have all their faces removed
    #    if not len(ngraph[p]):
    #        continue

        if np.sum(faces_mask[p]) == 0:
            continue

#        # grab pointers to voronoi vertices and to neighbors
#        # a face is just pointers to voronoi vertices
#        faces     = fgraph[p]
#        neighbors = ngraph[p]
#
#        # go through neighbors and corresponding faces
#        for i, neighbor in enumerate(neighbors):
#            
#            # grab voronoi vertices of face corresponding to neighbor
#            vor_verts  = voronoi_vertices[np.asarray(faces[i])]
#
#            # creat vector from one voronoi vertex to the other
#            # since vertices are ordered counter clockwise
#            normal = vor_verts[1] - vor_verts[0]
#
#            # area of face
#            area = np.sqrt(normal.dot(normal))
#
#            # rotate by -90 degress to create face normal
#            x, y = normal
#
#            # create vector form origin particle to center mass of face
#            dr = np.mean(vor_verts, axis=0) - particles[p]
#            x1, y1 = dr
#
#            if x1*y - y1*x > 0.0: x, y = y, -x
#            else: x, y = -y, x
#
#            # find the angle of the face
#            theta = np.angle(x+1j*y)
#
#            # find the velocity of the face
#            rl = particles[p]; rr = particles[neighbor]
#            wl = w[:,p]; wr = w[:,neighbor]
#
#            f = np.mean(vor_verts, axis=0)  # center mass of face
#
#            w_face = 0.5*(wl + wr)
#            w_face += np.sum((wl - wr)*(f-(rr + rl)*0.5))*(rr-rl)/np.sum((rr-rl)**2)
#
#            w_face_x, w_face_y = w_face
#
#            # store the angle area and points left and right of face
#            face_list.append([theta, area, w_face_x, w_face_y, p, neighbor])



        # indices of voronoi vertices that make up the faces
        voronoi_faces = np.array(face_graph[p])[faces_mask[p]]
        normal = voronoi_vertices[voronoi_faces]

        # center of mass for each face
        f = np.mean(normal, axis=1)

        # normal vector for each face
        normal = normal[:,0,:] - normal[:,1,:]

        # rotate normal vector by -90 degrees
        normal[:,[0, 1]] = normal[:, [1, 0]]
        normal[:,1] *= -1

        # vector from origin particle to center mass of faces
        dr = f - particles[p]

        # dot product to find which direction the normal is facing
        i = np.sum(normal*dr, axis=1) < 0.0

        # change the direction of normal to face away of left particle
        normal[i,:] *= -1

        # find the angle of the normal
        theta = np.angle(normal[:,0] + 1j*normal[:,1])

        # calculate the area of each face
        area = (normal*normal).sum(axis=1)
        np.sqrt(area, area)

        # find the velocity of the face
        neighbors = np.asarray(neighbor_graph[p])[faces_mask[p]]

        # particle position and velocity left and right to face
        rl = particles[p]
        rr = particles[neighbors]
        wl = w[:,p]
        wr = w[:,neighbors].T

        w_face = 0.5*(wl + wr)
        w_face += np.sum((wl - wr)*(f-(rr + rl)*0.5))*(rr-rl)/np.sum((rr-rl)**2)

        #import pdb;pdb.set_trace()

        k_new = k_old + neighbors.size
        j_ind = np.arange(k_old, k_new)

        faces_info[0,j_ind] = theta
        faces_info[1,j_ind] = area
        faces_info[2,j_ind] = w_face[:,0]
        faces_info[3,j_ind] = w_face[:,1]
        faces_info[4,j_ind] = np.ones(neighbors.size)*p
        faces_info[5,j_ind] = neighbors

        k_old = k_new

    return faces_info




#        # destroy link of neighbors to point 
#        for neighbor in neighbors:
#
#            k = ngraph[neighbor].index(p)
#
#            ngraph[neighbor].pop(k)
#            fgraph[neighbor].pop(k)
#
#        # destroy link of point to neighbors
#        ngraph[p] = []
#        fgraph[p] = []
#
#
#    ans =  np.transpose(np.asarray(face_list))
#    import pdb;pdb.set_trace()
#
#    #return np.transpose(np.asarray(face_list))
#    return ans
