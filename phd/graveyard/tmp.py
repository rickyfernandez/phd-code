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

    part = []
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

    tag = pc['tag']
    ghost = tag == 1
    plt.scatter(pc['position-x'][ghost], pc['position-y'][ghost], color='lightsteelblue')

    plt.savefig("plot_init_proc_%d.pdf" % rank, format='pdf')
    plt.clf()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:

    L = 1.  # domain size
    n = 50  # number of points per dimension

    # generate the domain including ghost particles
    dx = L/n
    xs = np.arange(n, dtype=np.float64)*dx + 0.5*dx
    ys = np.arange(n, dtype=np.float64)*dx + 0.5*dx

    X, Y = np.meshgrid(xs,ys); Y = np.flipud(Y)
    xs = X.flatten(); ys = Y.flatten()

    ids = np.arange(xs.size, dtype=np.int32)

    # let root processor create the data
    x = np.arange(xs.size, dtype=np.int32)
    lengths = np.split(x, size)
    lengths = [sub.size for sub in lengths]

    send = np.copy(lengths)

    disp = np.zeros(size, dtype=np.int32)
    for i in range(1,size):
        disp[i] = lengths[i-1] + disp[i-1]
else:
    xs = None
    ys = None
    ids = None
    lengths = None
    disp = None
    send = None

send = comm.scatter(send, root=0)
xlocal = np.empty(send,dtype=np.float64)
ylocal = np.empty(send,dtype=np.float64)
idlocal = np.empty(send,dtype=np.int32)

comm.Scatterv( [xs, (lengths, disp)], xlocal)
comm.Scatterv( [ys, (lengths, disp)], ylocal)
comm.Scatterv( [ids, (lengths, disp)], idlocal)

pc = ParticleContainer(send)
pc['position-x'][:] = xlocal
pc['position-y'][:] = ylocal
pc['process'][:] = rank
pc['type'][:] = ParticleTAGS.Undefined

#pc.register_property(send, "ids", "long")
pc["ids"][:] = idlocal

# perform the load decomposition
order = 21
dom = DomainLimits(dim=2, xmin=0., xmax=1.)
load_b = LoadBalance(pc, dom, comm=comm, order=order)
load_b.decomposition()

mesh = VoronoiMesh2D(pc)
bound = MultiCoreBoundary()
bound.create_ghost_particles(pc, mesh, dom, load_b, comm)
mesh.tessellate()
mesh.update_boundary_particles()
mesh.update_second_boundary_particles()
mesh.compute_cell_info()
vor_plot(pc, mesh, rank)
#
