import numpy as np
import copy

def faces_for_flux(particles, w, particles_index, neighbor_graph, face_graph, voronoi_vertices):

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
