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

    Lx = 1.    # domain size in x
    nx = 100
    n = nx*nx  # number of points

    dx = Lx/nx
    qx = np.arange(nx, dtype=np.float64)*dx + 0.5*dx
    x = np.zeros(n)
    y = np.zeros(n)

    part = 0
    for i in range(nx):
        for j in range(nx):
            x[part] = qx[i]
            y[part] = qx[j]
            part += 1

    # create particle container
    pc = ParticleContainer(n)
    pc['position-x'][:] = x
    pc['position-y'][:] = y
    pc['ids'][:] = np.arange(n)
    pc['tag'][:] = ParticleTAGS.Real

    # set ambient values
    pc['density'][:]  = 1.0  # density
    pc['pressure'][:] = 1.0  # total energy

    cells = pc['position-y'] > .5
    pc['density'][cells] = 0.125
    pc['pressure'][cells] = 0.1

    # zero out the velocities and particle type
    pc['velocity-x'][:] = 0.0
    pc['velocity-y'][:] = 0.0

    return pc

# create inital state of the simulation
pc = create_particles(1.4)

# create domain limits
domain = DomainLimits(dim=2, xmin=0., xmax=1.)

# boundary condition - periodic condition
boundary = Boundary(domain, boundary_type=BoundaryType.Reflective)

# tesselation algorithm
mesh = Mesh(boundary)

# reconstruction
reconstruction = PieceWiseConstant()

# riemann solver
riemann = HLL(reconstruction, gamma=1.4, cfl=0.3)

# integrator 
integrator = MovingMesh(pc, mesh, riemann, regularize=1)

# setup the solver
solver = Solver(integrator, tf=0.15, pfreq=1, relax_num_iterations=0, output_relax=False,
        fname='sod_2d_cgal')
solver.solve()
