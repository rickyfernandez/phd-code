from matplotlib.collections import LineCollection, PolyCollection
from ParentDirectory.Initialize.Initialize import GenerateInitialMesh
from ParentDirectory.Geometry.Geometry import Tesselation
from ParentDirectory.Geometry.Volume import Volume
from ParentDirectory.Boundary.periodic_no_move import PeriodicBoundary
from ParentDirectory.Boundary.Reflect import ReflectBoundary
from ParentDirectory.Geometry.FacesForFlux import FacesForFlux
from ParentDirectory.Reconstruction.MeshRegularization import MeshRegularization
from Sod import Sod
import matplotlib.pyplot as plt
import matplotlib.delaunay
import numpy as np
import random
import sys

problem = int(sys.argv[1])

if problem == 1:

    # plot all interior voronoi points 
    boundary = {"left":0.0, "right":1.0, "bottom":0.0, "top":1.0}
    particles, particles_index = GenerateInitialMesh(boundary, 100)

    neighbor_graph, face_graph, circum_centers = Tesselation(particles)


    l = []
    for i in particles_index["real"]:
        verts_indices = np.unique(np.asarray(face_graph[i]).flatten())
        verts = circum_centers[verts_indices]

        # coordinates of neighbors relative to particle p
        xc = verts[:,0] - particles[i,0]
        yc = verts[:,1] - particles[i,1]

        # sort in counter clock wise order
        sorted_vertices = np.argsort(np.angle(xc+1j*yc))
        verts = verts[sorted_vertices]
        l.append(verts)


    poly = PolyCollection(l, edgecolor='k')
    plt.gca().add_collection(poly)

    particles = PeriodicBoundary(particles, particles_index, neighbor_graph, boundary)
    ghost_indices = particles_index["ghost"]
    plt.plot(particles[ghost_indices,0], particles[ghost_indices,1], 'ro')

    real_indices = particles_index["real"]
    plt.plot(particles[real_indices,0], particles[real_indices,1], 'ro')

    # go throught ghost particles and plot them and the real particle it was created from
    ghost_map = particles_index["ghost_map"]
    for p in ghost_indices:

        ghost_point = particles[p]
        plt.text(ghost_point[0], ghost_point[1], '%d' % ghost_map[p], color='b', ha='center')

        real_point = particles[ghost_map[p]]
        plt.text(real_point[0], real_point[1]+0, '%d' % ghost_map[p], color='k', ha='center')

    plt.show()

if problem == 2:

    x  = np.linspace(0, 1, 20)
    dx = np.mean(np.diff(x))

    left = -dx*np.arange(1,4)[::-1]; x = np.append(left, x)
    right = 1.0+dx*np.arange(1,4); x = np.append(x, right)

    X, Y = np.meshgrid(x,x); Y = np.flipud(Y)
    x = X.flatten(); y = Y.flatten()

    i = np.where((0.25 < x) & (x < 0.75))[0]
    j = np.where((0.25 < y) & (y < 0.75))[0]
    k = np.intersect1d(i, j)

    x[k] += 0.2*dx*(2.0*np.random.random(len(k))-1.0)
    y[k] += 0.2*dx*(2.0*np.random.random(len(k))-1.0)

    particles = np.asarray(zip(x,y))

    neighbor_graph, face_graph, circum_centers = Tesselation(particles)

    # find particles in a known volume
    i = np.where((-0.5*dx < x) & (x < (1+0.5*dx)))[0]
    j = np.where((-0.5*dx < y) & (y < (1+0.5*dx)))[0]
    k = np.intersect1d(i, j)
    
    particles_index = {"real": k}
    # calculate square volume by summing voronoi volumes
    volume = Volume(particles, neighbor_graph, particles_index, face_graph, circum_centers)

    print "Square volume by voronoi cells:", np.sum(volume[0,:])
    print "True volume:", (1+dx)**2
    
    # plot particles
    plt.plot(particles[:,0], particles[:,1], 'ro', alpha=0.6)

    # plot center of mass
    plt.plot(volume[1,:], volume[2,:], 'ko', alpha=0.8)

    l = []
    for i in k:
        verts_indices = np.unique(np.asarray(face_graph[i]).flatten())
        verts = circum_centers[verts_indices]

        # coordinates of neighbors relative to particle p
        xc = verts[:,0] - particles[i,0]
        yc = verts[:,1] - particles[i,1]

        # sort in counter clock wise order
        sorted_vertices = np.argsort(np.angle(xc+1j*yc))
        verts = verts[sorted_vertices]
        l.append(verts)

#    l = []
#    for i in k:
#        verts = circum_centers[np.asarray(face_graph[i])[:,0]]
#        l.append(verts)

    poly = PolyCollection(l, edgecolor='k')
    plt.gca().add_collection(poly)
    plt.show()

if problem == 3:

    # plot all interior voronoi points 
    boundary = {"left":0.0, "right":1.0, "bottom":0.0, "top":1.0}
    particles, particles_index = GenerateInitialMesh(boundary, 100)

    neighbor_graph, face_graph, circum_centers = Tesselation(particles)
    #plt.plot(particles[:,0], particles[:,1], 'ro')

    # randomly pick an interior point and plot voronoi cell and neighbors
    points = particles_index["real"]
    point  = random.choice(points)

    faces = np.asarray(face_graph[point])
    neighbors = np.asarray(neighbor_graph[point])

    # plot face normals
    for i, p in enumerate(neighbors):

        face =  circum_centers[faces[i]]
        norm = face[1] - face[0]

        #plt.plot(face[0,0], face[0,1], 'bx')
        #plt.plot(face[1,0], face[1,1], 'rx')

        #rotate by -90 degress
        x, y = norm
        norm = np.array([y, -x])

        # center of face
        center = face.mean(axis=0)
        x0, y0 = center
        x1, y1 =  norm
        plt.arrow(x0, y0, x1, y1)


    vert_indices = np.asarray(face_graph[point])

    # plot neighbors
    for i, vi in enumerate(vert_indices):

        # center of edge
        p = circum_centers[vi].mean(axis=0)
        plt.text(p[0], p[1], '%d' % i, color='b', ha='center')

    # now plot neighbors
    neig_indices = np.asarray(neighbor_graph[point])
    for i, ni in enumerate(neig_indices):
        plt.text(particles[ni,0], particles[ni,1], '%d' % i, color='k', ha='center')

    verts = circum_centers[vert_indices]
    poly = PolyCollection(verts, edgecolor='k')
    plt.gca().add_collection(poly)
    plt.xlim(particles[point,0]-0.2, particles[point,0]+0.2)
    plt.ylim(particles[point,1]-0.2, particles[point,1]+0.2)
    plt.show()

if problem == 4:

    # plot all interior voronoi points 
    boundary = {"left":0.0, "right":1.0, "bottom":0.0, "top":1.0}
    particles, particles_index = GenerateInitialMesh(boundary, 100)

    neighbor_graph, face_graph, circum_centers = Tesselation(particles)


    l = []
    for i in particles_index["real"]:
        verts = circum_centers[np.asarray(face_graph[i])[:,0]]
        l.append(verts)


    poly = PolyCollection(l, edgecolor='k')
    plt.gca().add_collection(poly)

    particles = ReflectBoundary(particles, particles_index, neighbor_graph, boundary)
    ghost_indices = particles_index["ghost"]
    plt.plot(particles[ghost_indices,0], particles[ghost_indices,1], 'ro')

    real_indices = particles_index["real"]
    plt.plot(particles[real_indices,0], particles[real_indices,1], 'ro')

    # go throught ghost particles and plot them and the real particle it was created from
    ghost_map = particles_index["ghost_map"]
    for p in ghost_indices:

        ghost_point = particles[p]
        plt.text(ghost_point[0], ghost_point[1], '%d' % ghost_map[p], color='b', ha='center')

        real_point = particles[ghost_map[p]]
        plt.text(real_point[0], real_point[1]+0, '%d' % ghost_map[p], color='k', ha='center')

    plt.show()

if problem == 5:

    # plot all interior voronoi points 
    boundary = {"left":0.0, "right":1.0, "bottom":0.0, "top":1.0}
    particles, particles_index = GenerateInitialMesh(boundary, 100)

    neighbor_graph, face_graph, circum_centers = Tesselation(particles)

    particles = ReflectBoundary(particles, particles_index, neighbor_graph, boundary)
    neighbor_graph, face_graph, circum_centers = Tesselation(particles)

    l = []
    for i in particles_index["real"]:
        verts = circum_centers[np.asarray(face_graph[i])[:,0]]
        l.append(verts)

    poly = PolyCollection(l, edgecolor='k')
    plt.gca().add_collection(poly)

    ghost_indices = particles_index["ghost"]
    plt.plot(particles[ghost_indices,0], particles[ghost_indices,1], 'ro')

    real_indices = particles_index["real"]
    plt.plot(particles[real_indices,0], particles[real_indices,1], 'ro')

    # go throught ghost particles and plot them and the real particle it was created from
    ghost_map = particles_index["ghost_map"]
    for p in ghost_indices:

        ghost_point = particles[p]
        plt.text(ghost_point[0], ghost_point[1], '%d' % ghost_map[p], color='b', ha='center')

        real_point = particles[ghost_map[p]]
        plt.text(real_point[0], real_point[1]+0, '%d' % ghost_map[p], color='k', ha='center')

    plt.show()
    
if problem == 6:

    # plot all interior voronoi points 
    boundary = {"left":0.0, "right":1.0, "bottom":0.0, "top":1.0}
    particles, particles_index = GenerateInitialMesh(boundary, 20)

    neighbor_graph, face_graph, voronoi_vertices = Tesselation(particles)

    particles = ReflectBoundary(particles, particles_index, neighbor_graph, boundary)
    neighbor_graph, face_graph, voronoi_vertices = Tesselation(particles)


    faces_info = FacesForFlux(particles, particles_index, neighbor_graph, face_graph, voronoi_vertices)

    print faces_info[2::,:]

    # randomly pick a face 
    #face  = random.choice(np.arange(faces_info.shape[1]))
    #v = faces_info[:,face]
    
    for i in np.arange(faces_info.shape[1]):

        v = faces_info[:,i]
        p = particles[v[2::].astype(int)]
        plt.plot(p[:,0], p[:,1], 'm')


    l = []
    for i in particles_index["real"]:
        verts = voronoi_vertices[np.asarray(face_graph[i])[:,0]]
        l.append(verts)

    poly = PolyCollection(l, edgecolor='k')
    plt.gca().add_collection(poly)

    ghost_indices = particles_index["ghost"]
    plt.plot(particles[ghost_indices,0], particles[ghost_indices,1], 'ro')

    real_indices = particles_index["real"]
    plt.plot(particles[real_indices,0], particles[real_indices,1], 'ro')

    for p in particles_index["real"]:

        plt.text(particles[p,0], particles[p,1], '%d' % p, color='r', ha='center')

    for p in particles_index["ghost"]:

        plt.text(particles[p,0], particles[p,1], '%d' % p, color='r', ha='center')

    plt.show()

if problem == 7:

    x  = np.linspace(0, 1, 20)
    dx = np.mean(np.diff(x))

    left = -dx*np.arange(1,4)[::-1]; x = np.append(left, x)
    right = 1.0+dx*np.arange(1,4); x = np.append(x, right)

    X, Y = np.meshgrid(x,x); Y = np.flipud(Y)
    x = X.flatten(); y = Y.flatten()

    i = np.where((0.25 < x) & (x < 0.75))[0]
    j = np.where((0.25 < y) & (y < 0.75))[0]
    k = np.intersect1d(i, j)

    x[k] += 0.5*dx*(2.0*np.random.random(len(k))-1.0)
    y[k] += 0.5*dx*(2.0*np.random.random(len(k))-1.0)

    particles = np.asarray(zip(x,y))

    neighbor_graph, face_graph, circum_centers = Tesselation(particles)

    # find particles in a known volume
    i = np.where((-0.5*dx < x) & (x < (1+0.5*dx)))[0]
    j = np.where((-0.5*dx < y) & (y < (1+0.5*dx)))[0]
    k = np.intersect1d(i, j)
    
    particles_index = {"real": k}
    # calculate square volume by summing voronoi volumes
    vol_center_mass = Volume(particles, neighbor_graph, particles_index, face_graph, circum_centers)

    # make fake data
    data = np.zeros((4, particles.shape[0]))
    data[0,:] = 1.0 # constant density
    data[3,:] = 1.0 # constant pressure

    for j in range(10): # relaxation steps
    
        # plot particles
        plt.plot(particles[k,0], particles[k,1], 'ro', alpha=0.6)

        # plot center of mass
        plt.plot(vol_center_mass[1,:], vol_center_mass[2,:], 'ko', alpha=0.8)

        l = []
        for i in k:
            verts = circum_centers[np.asarray(face_graph[i])[:,0]]
            l.append(verts)

        poly = PolyCollection(l, edgecolor='k')
        plt.gca().add_collection(poly)
        plt.savefig("regulization_" + `j`.zfill(4))
        plt.clf()

        w = MeshRegularization(data, particles, 1.4, vol_center_mass, particles_index)
        particles[k] += 0.01*np.transpose(w)
        neighbor_graph, face_graph, circum_centers = Tesselation(particles)
        vol_center_mass = Volume(particles, neighbor_graph, particles_index, face_graph, circum_centers)
