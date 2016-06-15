from mpi4py import MPI
import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle

from utils.particle_tags import ParticleTAGS
from domain.domain import DomainLimits
from load_balance.load_balance import LoadBalance
from containers.containers import ParticleContainer
from boundary.boundary import MultiCoreBoundary
from mesh.voronoi_mesh import VoronoiMesh2D

def vor_plot(pc, mesh, rank):

    # debugging plot --- turn to a routine later ---
    l = []
    ii = 0; jj = 0
    for i in range(pc.get_number_of_particles()):

        jj += mesh['number of neighbors'][i]*2

        if pc['tag'][i] == ParticleTAGS.Real or pc['type'][i] == ParticleTAGS.Boundary or pc['type'][i] == ParticleTAGS.BoundarySecond:

            verts_indices = np.unique(mesh['faces'][ii:jj])
            verts = mesh['voronoi vertices'][verts_indices]

            # coordinates of neighbors relative to particle p
            xc = verts[:,0] - pc['position-x'][i]
            yc = verts[:,1] - pc['position-y'][i]

            # sort in counter clock wise order
            sorted_vertices = np.argsort(np.angle(xc+1j*yc))
            verts = verts[sorted_vertices]

            l.append(Polygon(verts, True))

        ii = jj

    colors = []
    for i in range(pc.get_number_of_particles()):
        if pc['tag'][i] == ParticleTAGS.Real:
            colors.append(0.25)
        if pc['type'][i] == ParticleTAGS.Boundary:
            colors.append(0.5)
        if pc['type'][i] == ParticleTAGS.BoundarySecond:
            colors.append(0.75)

    fig, ax = plt.subplots()
    p = PatchCollection(l, alpha=0.4)
    p.set_array(np.array(colors))
    p.set_clim([0.1, 1.0])

    ax.set_xlim(-.1,1.1)
    ax.set_ylim(-.1,1.1)
    ax.add_collection(p)

#    second = pc['type'] == 7
#    print 'rank:', rank, np.sum(second)
#    plt.scatter(pa['position-x'][second], pa['position-y'][second], color='lightsteelblue')

    plt.savefig("plot_init_proc_%d.pdf" % rank, format='pdf')
    plt.clf()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# generate random particles in a unit box for each process
np.random.seed(rank)
my_num_particles = np.random.randint(64, 256)
#my_num_particles = np.random.randint(512, 1024)
my_particles = np.random.random(2*my_num_particles).reshape(2, my_num_particles).astype(np.float64)
#my_particles = np.random.normal(0.5, 0.1, 2*my_num_particles).reshape(2, my_num_particles).astype(np.float64)

#pa = ParticleArray(my_num_particles)
pa = ParticleContainer(my_num_particles)
pa['position-x'][:] = my_particles[0,:]
pa['position-y'][:] = my_particles[1,:]
pa['process'][:] = rank
pa['type'][:] = ParticleTAGS.Undefined

#plot initial distribuition
#plt.scatter(pc['position-x'], pc['position-y'])
#plt.savefig("plot_init_proc_%d.png" % rank)
#plt.clf()

# perform load balance
#lb = LoadBalance(pc, comm=comm)
#global_tree = lb.decomposition()

# perform the load decomposition
order = 21
dom = DomainLimits(dim=2, xmin=0., xmax=1.)
load_b = LoadBalance(pa, dom, comm=comm, order=order)
load_b.decomposition()
#global_tree = load_b.decomposition()

#num_particles = pa.get_num_real_particles()

#print 'rank: %d number of particles %d: number of leaves %d' % (rank, pc.num_real_particles, lb.global_work.size)













## create border particles
#border = lb.create_ghost_particles(global_tree)
#
## plot particles
#plt.scatter(pc['position-x'][:num_particles], pc['position-y'][:num_particles])
##plt.scatter(pc['position-x'][num_particles:], pc['position-y'][num_particles:], color="red")
##plt.scatter(pc['position-x'][border], pc['position-y'][border], color="green")
##plt.scatter(send_data['position-x'], send_data['position-y'], color="red")
##plt.xlim(-0.2,1.2)
##plt.ylim(-0.2,1.2)
#plt.savefig("plot_proc_%d.png" % rank)
#


# plot particles
#global_tree.create_boundary_particles(pa, rank, load_b.leaf_proc)
mesh = VoronoiMesh2D(pa)
bound = MultiCoreBoundary()
bound.create_ghost_particles(pa, mesh, dom, load_b, comm)
mesh.tessellate()
mesh.update_boundary_particles()
mesh.update_second_boundary_particles()
vor_plot(pa, mesh, rank)
#
#
#
#
#
##tag = pa['tag']
##real = tag == 0
##plt.scatter(pa['position-x'][real], pa['position-y'][real], color='lightsteelblue')
##
### plot global tree
#current_axis = plt.gca()
#for xn, yn, wn in load_b.global_tree.dump_data():
#    x = xn/2.0**order
#    y = yn/2.0**order
#    w = wn/2.0**order
#    current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, fill=None))
##
#tag = pa['tag']
#real  = tag == 0
#ghost = tag == 1
#first = tag == 8
#print "first", sum(first)
#print "rank:", rank, "number of ghost:", sum(ghost)
#plt.scatter(pa['position-x'][real], pa['position-y'][real], color='lightsteelblue')
#plt.scatter(pa['position-x'][first], pa['position-y'][first], color='red')
##corner = pa['type'] == 10
##plt.scatter(pa['position-x'][corner], pa['position-y'][corner], color="blue")
##exterior = pa['type'] == 11
##plt.scatter(pa['position-x'][exterior], pa['position-y'][exterior], color="red")
##interior = pa['type'] == 12
##plt.scatter(pa['position-x'][interior], pa['position-y'][interior], color="green")
##
#plt.xlim(-0.2,1.2)
#plt.ylim(-0.2,1.2)
#plt.savefig("plot_proc_%d.png" % rank)
##plt.clf()
#
#
#
#
#
#
#
##
### plot all the particles
##sendbuf = np.array([my_num_particles], dtype=np.int32)
##counts = np.empty(size, dtype=np.int32)
##
##comm.Allgather(sendbuf=sendbuf, recvbuf=counts)
##
##global_tot = np.sum(counts)
##offsets = np.zeros(size)
##offsets[1:] = np.cumsum(counts)[:-1]
##
##X = np.empty(global_tot, dtype=np.float64)
##Y = np.empty(global_tot, dtype=np.float64)
##
##x = my_particles[0,:]
##y = my_particles[1,:]
##
##comm.Allgatherv(sendbuf=x, recvbuf=[X, counts, offsets, MPI.DOUBLE])
##comm.Allgatherv(sendbuf=y, recvbuf=[Y, counts, offsets, MPI.DOUBLE])
###plt.scatter(X, Y, color="red")
##
##
##mesh = VoronoiMesh2D()
##p = np.array([X, Y])
###graphs = mesh.tessellate(p)
##mesh.tessellate(p)
##l = []
##ii = 0; jj = 0
##for ip in xrange(global_tot):
##
##    jj += mesh["number of neighbors"][ip] *2
##    x_tmp = X[ip]
##    y_tmp = Y[ip]
##
##    if 0.05 < x_tmp < 0.95 and 0.05 < y_tmp < 0.95:
##        verts_indices = np.unique(mesh["faces"][ii:jj])
##        verts = mesh["voronoi vertices"][verts_indices]
##
##        xc = verts[:,0] - X[ip]
##        yc = verts[:,1] - Y[ip]
##
##        # sort in counter clock wise order
##        sorted_vertices = np.argsort(np.angle(xc+1j*yc))
##        verts = verts[sorted_vertices]
##
##        l.append(Polygon(verts, True))
##
##    ii = jj
##
##
##p = PatchCollection(l, alpha=0.1)
##current_axis.add_collection(p)
##
##
### debugging plot --- turn to a routine later ---
##mesh = VoronoiMesh2D()
##p = np.array([pc['position-x'], pc['position-y']])
###graphs = mesh.tessellate(p)
##mesh.tessellate(p)
##l = []
##ii = 0; jj = 0
##for ip in xrange(0,pc.num_real_particles):
##
##    jj += mesh["number of neighbors"][ip] *2
##    verts_indices = np.unique(mesh["faces"][ii:jj])
##    verts = mesh["voronoi vertices"][verts_indices]
##
##    xc = verts[:,0] - pc["position-x"][ip]
##    yc = verts[:,1] - pc["position-y"][ip]
##
##    # sort in counter clock wise order
##    sorted_vertices = np.argsort(np.angle(xc+1j*yc))
##    verts = verts[sorted_vertices]
##
##    l.append(Polygon(verts, True))
##
##    ii = jj
##
##
###cells = self.particles_index["real"]
###dens = self.fields.get_field("density")
###velx = self.fields.get_field("velocity-x")
###vely = self.fields.get_field("velocity-y")
###pres = self.fields.get_field("pressure")
##
### add colormap
###colors = []
###for i in self.particles_index["real"]:
###    colors.append(dens[i])
##
###fig, ax = plt.subplots(figsize=(8, 8))
###fig, ax = plt.subplots()
##p = PatchCollection(l, facecolors="#0099FF")
###p.set_array(np.array(colors))
###p.set_clim([0, 4])
##
###ax.set_xlim(-0.2,1.2)
###ax.set_ylim(-0.2,1.2)
###ax.set_aspect(2)
###ax.add_collection(p)
##current_axis.add_collection(p)
##
###plt.colorbar(p, orientation='horizontal')
###plt.colorbar(p)
##plt.savefig("mesh_proc_%d.png" % rank)
##plt.clf()
