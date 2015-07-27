from mpi4py import MPI
import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle

from domain.domain import DomainLimits
from load_balance.load_balance import LoadBalance
from particles.particle_array import ParticleArray
from boundary.boundary import Boundary
#from mesh.voronoi_mesh_2d import VoronoiMesh2D

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# generate random particles in a unit box for each process
np.random.seed(rank)
#my_num_particles = np.random.randint(64, 256)
my_num_particles = np.random.randint(256, 512)
my_particles = np.random.random(2*my_num_particles).reshape(2, my_num_particles).astype(np.float64)
#my_particles = np.random.normal(0.5, 0.1, 2*my_num_particles).reshape(2, my_num_particles).astype(np.float64)

pa = ParticleArray(my_num_particles)
pa['position-x'][:] = my_particles[0,:]
pa['position-y'][:] = my_particles[1,:]
pa['process'][:] = rank

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
global_tree = load_b.decomposition()

#num_particles = pa.get_num_real_particles()

#print 'rank: %d number of particles %d: number of leaves %d' % (rank, pc.num_real_particles, lb.global_work.size)

#lb.create_ghost_particles()
current_axis = plt.gca()
# plot bounding leaf
#for node in p:
#    x = node[0]/2.0**order
#    y = node[1]/2.0**order
##    w = node[2]/2.0**order
#    plt.plot(x, y, 'r*')
##    current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, color='orange', alpha=0.5))
#
# plot global tree
current_axis = plt.gca()
for node in global_tree.dump_data():
    x = node[0]/2.0**order
    y = node[1]/2.0**order
    w = node[2]/2.0**order
    current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, fill=None))
#
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
bound = Boundary()
bound.create_ghost_particles(pa, load_b.leaf_proc, global_tree, dom, comm)
#print pa['tag']
tag = pa['tag']
real = tag == 0
plt.scatter(pa['position-x'][real], pa['position-y'][real])
plt.scatter(pa['position-x'][~real], pa['position-y'][~real], color="red")
##plt.scatter(pc['position-x'][border], pc['position-y'][border], color="green")
##plt.scatter(send_data['position-x'], send_data['position-y'], color="red")
plt.xlim(-0.2,1.2)
plt.ylim(-0.2,1.2)
plt.savefig("plot_proc_%d.png" % rank)
#
## plot all the particles
#sendbuf = np.array([my_num_particles], dtype=np.int32)
#counts = np.empty(size, dtype=np.int32)
#
#comm.Allgather(sendbuf=sendbuf, recvbuf=counts)
#
#global_tot = np.sum(counts)
#offsets = np.zeros(size)
#offsets[1:] = np.cumsum(counts)[:-1]
#
#X = np.empty(global_tot, dtype=np.float64)
#Y = np.empty(global_tot, dtype=np.float64)
#
#x = my_particles[0,:]
#y = my_particles[1,:]
#
#comm.Allgatherv(sendbuf=x, recvbuf=[X, counts, offsets, MPI.DOUBLE])
#comm.Allgatherv(sendbuf=y, recvbuf=[Y, counts, offsets, MPI.DOUBLE])
##plt.scatter(X, Y, color="red")
#
#
#mesh = VoronoiMesh2D()
#p = np.array([X, Y])
##graphs = mesh.tessellate(p)
#mesh.tessellate(p)
#l = []
#ii = 0; jj = 0
#for ip in xrange(global_tot):
#
#    jj += mesh["number of neighbors"][ip] *2
#    x_tmp = X[ip]
#    y_tmp = Y[ip]
#
#    if 0.05 < x_tmp < 0.95 and 0.05 < y_tmp < 0.95:
#        verts_indices = np.unique(mesh["faces"][ii:jj])
#        verts = mesh["voronoi vertices"][verts_indices]
#
#        xc = verts[:,0] - X[ip]
#        yc = verts[:,1] - Y[ip]
#
#        # sort in counter clock wise order
#        sorted_vertices = np.argsort(np.angle(xc+1j*yc))
#        verts = verts[sorted_vertices]
#
#        l.append(Polygon(verts, True))
#
#    ii = jj
#
#
#p = PatchCollection(l, alpha=0.1)
#current_axis.add_collection(p)
#
#
## debugging plot --- turn to a routine later ---
#mesh = VoronoiMesh2D()
#p = np.array([pc['position-x'], pc['position-y']])
##graphs = mesh.tessellate(p)
#mesh.tessellate(p)
#l = []
#ii = 0; jj = 0
#for ip in xrange(0,pc.num_real_particles):
#
#    jj += mesh["number of neighbors"][ip] *2
#    verts_indices = np.unique(mesh["faces"][ii:jj])
#    verts = mesh["voronoi vertices"][verts_indices]
#
#    xc = verts[:,0] - pc["position-x"][ip]
#    yc = verts[:,1] - pc["position-y"][ip]
#
#    # sort in counter clock wise order
#    sorted_vertices = np.argsort(np.angle(xc+1j*yc))
#    verts = verts[sorted_vertices]
#
#    l.append(Polygon(verts, True))
#
#    ii = jj
#
#
##cells = self.particles_index["real"]
##dens = self.fields.get_field("density")
##velx = self.fields.get_field("velocity-x")
##vely = self.fields.get_field("velocity-y")
##pres = self.fields.get_field("pressure")
#
## add colormap
##colors = []
##for i in self.particles_index["real"]:
##    colors.append(dens[i])
#
##fig, ax = plt.subplots(figsize=(8, 8))
##fig, ax = plt.subplots()
#p = PatchCollection(l, facecolors="#0099FF")
##p.set_array(np.array(colors))
##p.set_clim([0, 4])
#
##ax.set_xlim(-0.2,1.2)
##ax.set_ylim(-0.2,1.2)
##ax.set_aspect(2)
##ax.add_collection(p)
#current_axis.add_collection(p)
#
##plt.colorbar(p, orientation='horizontal')
##plt.colorbar(p)
#plt.savefig("mesh_proc_%d.png" % rank)
#plt.clf()
