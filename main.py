from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib

import moving_mesh.reconstruction as reconstruction
import moving_mesh.test_problems as test_problems
import moving_mesh.boundary as boundary
import moving_mesh.mesh as mesh
import moving_mesh.solvers as solvers


import numpy as np
import copy


# cfl condition
CFL = 0.5

# initial sod shock tube problem
data, particles, gamma, particles_index, boundary_dic = test_problems.sedov()

# make initial teselation
neighbor_graph, face_graph, voronoi_vertices = mesh.tessellation(particles)

# calculate volume of real particles 
volume = mesh.volume_center_mass(particles, neighbor_graph, particles_index, face_graph, voronoi_vertices)

# convert data to mass, momentum, and energy
reconstruction.conservative_variables(data, volume[0,:])

for k in range(10):

    print "from main: loop", k

    # generate periodic ghost particles with links to original real particles 
    particles = boundary.reflect(particles, particles_index, neighbor_graph, boundary_dic)

    # make tesselation returning graph of neighbors graph of faces and voronoi vertices
    neighbor_graph, face_graph, voronoi_vertices = mesh.tessellation(particles)

    # calculate volume of all real particles 
    volume = mesh.volume_center_mass(particles, neighbor_graph, particles_index, face_graph, voronoi_vertices)

    # calculate primitive variables for real particles
    primitive = reconstruction.primitive_variables(data, volume[0,:], gamma)

    # calculate global time step
    dt = reconstruction.time_step(primitive, volume[0,:], gamma, CFL)

    # copy values for ghost particles
    ghost_map = particles_index["ghost_map"]
    primitive = np.hstack((primitive,
        primitive[:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))

    # reverse velocities
    boundary.reverse_velocities_boundary(particles, primitive, particles_index, boundary_dic)

    # mesh regularization
    w = reconstruction.mesh_regularization(primitive, particles, gamma, volume, particles_index)
    w = np.hstack((w, w[:, np.asarray([ghost_map[i] for i in particles_index["ghost"]])]))

    # add particle velocities
    w[:, particles_index["real"]]  += primitive[1:3, particles_index["real"]]
    w[:, particles_index["ghost"]] += primitive[1:3, particles_index["ghost"]]

    # grab each face with particle id of the left and right particles as well angle and area
    faces_info = mesh.faces_for_flux(particles, w, particles_index, neighbor_graph, face_graph, voronoi_vertices)

    # grab left and right states
    left  = primitive[:, faces_info[4,:].astype(int)]
    right = primitive[:, faces_info[5,:].astype(int)]

    # calculate state at edge
    fluxes = solvers.pvrs(left, right, faces_info, gamma)

    # update conserved variables
    solvers.update(data, fluxes, dt, faces_info, particles_index)

    # move particles
    particles[particles_index["real"],:] += dt*np.transpose(w[:, particles_index["real"]])






    # plot data
    #plt.plot(particles[particles_index["real"],0], data[0, particles_index["real"]]/volume[0,:], 'xr')
    #plt.plot(particles[particles_index["real"],0], data[0, particles_index["real"]]/volume[0,:], 'xr')
    #plt.ylim(0.,1.1)
    #plt.savefig("scatter_sod_"+`k`.zfill(4))
    #plt.clf()
    
#if not k%10:
#    l = []
#    for i in particles_index["real"]:
#
#        verts_indices = np.unique(np.asarray(face_graph[i]).flatten())
#        verts = voronoi_vertices[verts_indices]
#
#        # coordinates of neighbors relative to particle p
#        xc = verts[:,0] - particles[i,0]
#        yc = verts[:,1] - particles[i,1]
#
#        # sort in counter clock wise order
#        sorted_vertices = np.argsort(np.angle(xc+1j*yc))
#        verts = verts[sorted_vertices]
#
#        l.append(Polygon(verts, True))
#
#    # add colormap
#    colors = []
#    for i in particles_index["real"]:
#        colors.append(data[0,i]/volume[0,i])
#
#    fig, ax = plt.subplots()
#    p = PatchCollection(l, cmap=matplotlib.cm.jet, edgecolors='none', alpha=0.4)
#    p.set_array(np.array(colors))
#    p.set_clim([0, 4.1])
#    ax.add_collection(p)
#    plt.colorbar(p)
#    plt.savefig("Sedov_"+`k`.zfill(4))
#    plt.clf()
