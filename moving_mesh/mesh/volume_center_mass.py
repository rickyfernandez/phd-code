import numpy as np

def cell_volume(particle_id, particles, neighbor_graph, face_graph, circum_centers):

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
    #import pdb;pdb.set_trace()

    return cell_volume, cm[0], cm[1]

def volume_center_mass(particles, neighbor_graph, particles_index, face_graph, voronoi_vertices):

    # calculate volume of real particles 
    vals = np.empty((3, particles_index["real"].shape[0]), dtype="float64")
    for i, particle_id in enumerate(particles_index["real"]):
        vals[:,i] = cell_volume(particle_id, particles, neighbor_graph, face_graph, voronoi_vertices)
    return vals
