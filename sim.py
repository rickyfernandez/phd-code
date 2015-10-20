from phd.solver.solver import Solver
from phd.riemann.riemann import HLLC, HLL
from phd.domain.domain import DomainLimits
from phd.integrate.integrator import MovingMesh
from phd.mesh.voronoi_mesh import VoronoiMesh2D
from phd.utils.particle_tags import ParticleTAGS
from phd.boundary.boundary import SingleCoreBoundary
from phd.containers.containers import ParticleContainer
from phd.reconstruction.reconstruction import PieceWiseConstant

import numpy as np

def create_particles():

    Lx = 1.  # domain size in x
    Ly = 1.  # domain size in y
    nx = 50  # number of points in x
    ny = 50  # number of points in y

    # generate the domain including ghost particles
    dx = Lx/nx
    dy = Ly/ny
    x = (np.arange(nx+6, dtype=np.float64) - 3)*dx + 0.5*dx
    y = (np.arange(ny+6, dtype=np.float64) - 3)*dy + 0.5*dy

    X, Y = np.meshgrid(x,y); Y = np.flipud(Y)
    x = X.flatten(); y = Y.flatten()

    # create particle container
    pc = ParticleContainer(x.size)
    pc['position-x'][:] = x
    pc['position-y'][:] = y

    # define domain boundaries 
    left   = 0.0; right = 1.0
    bottom = 0.0; top   = 1.0

    # find all particles in the left side 
    partition = x <= 0.5

    # set initial data on left side 
    pc['density'][partition]  = 1.0  # density
    pc['pressure'][partition] = 1.0  # total energy
    pc['tag'][partition] = ParticleTAGS.Real

    # set initial data on right side 
    pc['density'][~partition]  = 0.125  # density
    pc['pressure'][~partition] = 0.1    # total energy
    pc['tag'][~partition] = ParticleTAGS.Ghost

    indices = (((left <= x) & (x <= right)) & ((bottom <= y) & (y <= top)))
    pc['tag'][indices] = ParticleTAGS.Real
    pc['tag'][~indices] = ParticleTAGS.Ghost

    # zero out the velocities and particle type
    pc['velocity-x'][:] = 0.0
    pc['velocity-y'][:] = 0.0
    pc['type'][:] = ParticleTAGS.Undefined

    pc.align_particles()

    return pc

# create inital state of the simulation
pc = create_particles()

# create domain limits
domain = DomainLimits(dim=2, xmin=0., xmax=1.)

# boundary condition
boundary = SingleCoreBoundary()

# tesselation algorithm
mesh = VoronoiMesh2D(pc)

# reconstruction and riemann solver
reconstruction = PieceWiseConstant()
riemann = HLL(reconstruction, gamma=1.4, cfl=0.3)

# integrator 
integrator = MovingMesh(mesh, riemann)

# setup the solver
solver = Solver(mesh, integrator, boundary, domain, tf=0.15)
solver.solve()
