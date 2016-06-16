from phd.mesh.mesh import Mesh
from phd.riemann.riemann import HLL
from phd.solver.solver import Solver
from phd.domain.domain import DomainLimits
from phd.integrate.integrator import MovingMesh
from phd.utils.particle_tags import ParticleTAGS
from phd.containers.containers import ParticleContainer
from phd.boundary.boundary import Boundary, BoundaryType
from phd.reconstruction.reconstruction import PieceWiseConstant

import numpy as np

def create_particles(gamma):

    Lx = 1.  # domain size in x
    Ly = 1.  # domain size in y
    nx = 51  # number of points in x
    ny = 51  # number of points in y

    N = nx*ny

    # generate the domain particles dx = Lx/nx
    dx = Lx/nx
    dy = Ly/ny
    x = np.arange(nx, dtype=np.float64)*dx + 0.5*dx
    y = np.arange(ny, dtype=np.float64)*dy + 0.5*dy

    X, Y = np.meshgrid(x,y); Y = np.flipud(Y)
    x = X.flatten(); y = Y.flatten()

    # create particle container
    pc = ParticleContainer(x.size)
    pc['position-x'][:] = x
    pc['position-y'][:] = y

    #pc = ParticleContainer(N)
    #pc['position-x'][:] = np.random.random(N)
    #pc['position-y'][:] = np.random.random(N)
    pc['velocity-x'][:] = 0.0
    pc['velocity-y'][:] = 0.0

    pc['ids'][:] = np.arange(N)
    pc['tag'][:] = ParticleTAGS.Real

    # set ambient values
    pc['density'][:]  = 1.0               # density
    pc['pressure'][:] = 1.0E-5*(gamma-1)  # total energy

    r = 0.01
    cells = ((pc['position-x']-.5)**2 + (pc['position-y']-.5)**2) <= r**2
    #pc['pressure'][cells] = 1.0/(np.pi*r**2)*(gamma-1)
    pc['pressure'][cells] = 1.0/(dx*dx)*(gamma-1)

    return pc

# create inital state of the simulation
pc = create_particles(1.4)

# create domain limits
domain = DomainLimits(dim=2, xmin=0., xmax=1.)

# boundary condition
boundary = Boundary(domain, boundary_type=BoundaryType.Reflective)

# tesselation algorithm
mesh = Mesh(boundary)

# reconstruction
reconstruction = PieceWiseConstant()

# riemann solver
riemann = HLL(reconstruction, gamma=1.4, cfl=0.5)

# integrator 
integrator = MovingMesh(pc, mesh, riemann, regularize=1)

# setup the solver
#solver = Solver(integrator, tf=0.1, pfreq=1, relax_num_iterations=8, output_relax=False,
solver = Solver(integrator, tf=0.1, pfreq=1, relax_num_iterations=0, output_relax=False,
        fname='sedov_2d_cgal')
solver.solve()
