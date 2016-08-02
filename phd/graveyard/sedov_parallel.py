from phd.solver.solver import SolverParallel
from phd.riemann.riemann import HLLC, HLL
from phd.domain.domain import DomainLimits
from load_balance.load_balance import LoadBalance
from phd.integrate.integrator import MovingMesh
from phd.mesh.voronoi_mesh import VoronoiMesh2D
from phd.utils.particle_tags import ParticleTAGS
from phd.boundary.boundary import SingleCoreBoundary
from boundary.boundary import MultiCoreBoundary
from phd.containers.containers import ParticleContainer
from phd.reconstruction.reconstruction import PieceWiseConstant, PieceWiseLinear

from mpi4py import MPI
import numpy as np

def create_initial_state(pc, dx, gamma):

    x = pc['position-x']
    y = pc['position-y']

    # set ambient values
    pc['density'][:]  = 1.0               # density
    pc['pressure'][:] = 1.0E-5*(gamma-1)  # total energy

    r = 0.01
    cells = ((x-.5)**2 + (y-.5)**2) <= r**2
    pc['pressure'][cells] = 1.0/(dx*dx)*(gamma-1)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:

    L = 1.  # domain size
    n = 51  # number of points per dimension

    # generate the domain particles
    dx = L/n
    xs = np.arange(n, dtype=np.float64)*dx + 0.5*dx
    ys = np.arange(n, dtype=np.float64)*dx + 0.5*dx

    X, Y = np.meshgrid(xs,ys); Y = np.flipud(Y)
    xs = X.flatten(); ys = Y.flatten()

    # let root processor create the data
    pc_root = ParticleContainer(xs.size)
    pc_root['position-x'][:] = xs
    pc_root['position-y'][:] = ys
    pc_root['ids'][:] = np.arange(xs.size)

    create_initial_state(pc_root, dx, 1.4)

    x = np.arange(xs.size, dtype=np.int32)
    lengths = np.array_split(x, size)
    lengths = [sub.size for sub in lengths]

    send = np.copy(lengths)

    disp = np.zeros(size, dtype=np.int32)
    for i in range(1,size):
        disp[i] = lengths[i-1] + disp[i-1]

else:

    lengths = disp = send = None
    pc_root = {'position-x': None,
               'position-y': None,
               'density': None,
               'ids': None,
               'pressure': None}

# tell each processor how many particles it will hold
send = comm.scatter(send, root=0)

# allocate local particle container
pc = ParticleContainer(send)

# import particles from root
fields = ['position-x', 'position-y', 'density', 'pressure', 'ids']
for field in fields:
    comm.Scatterv([pc_root[field], (lengths, disp)], pc[field])

pc['velocity-x'][:] = 0.0
pc['velocity-y'][:] = 0.0
pc['process'][:] = rank
pc['tag'][:] = ParticleTAGS.Real
pc['type'][:] = ParticleTAGS.Undefined

# setup simulation classes
domain = DomainLimits(dim=2, xmin=0., xmax=1.)       # domain limits
boundary = MultiCoreBoundary()                       # boundary condition
mesh = VoronoiMesh2D(pc)                             # tesselation algorithm
#reconstruction = PieceWiseConstant()                 # reconstruction
reconstruction = PieceWiseLinear()                 # reconstruction
riemann = HLL(reconstruction, gamma=1.4, cfl=0.5)    # riemann solver
integrator = MovingMesh(mesh, riemann, regularize=1) # time update 
load_balance = LoadBalance(pc, domain, comm=comm)    # domain partioner 

# setup the solver
solver = SolverParallel(mesh, integrator, boundary, domain,
        load_balance, comm, tf=0.1, fname='sedov_parallel_simulation')
solver.solve()
