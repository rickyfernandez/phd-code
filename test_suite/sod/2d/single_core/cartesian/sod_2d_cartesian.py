import phd
import numpy as np

def create_particles(gamma):

    Lx = 1.    # domain size in x
    nx = 50    # particles per dim
    n = nx*nx  # number of points

    dx = Lx/nx # spacing between particles

    # create particle container
    pc = phd.ParticleContainer(n)
    part = 0
    for i in range(nx):
        for j in range(nx):
            pc['position-x'][part] = (i+0.5)*dx
            pc['position-y'][part] = (j+0.5)*dx
            pc['ids'][part] = part
            part += 1

    # set ambient values
    pc['density'][:]  = 1.0  # density
    pc['pressure'][:] = 1.0  # total energy

    cells = pc['position-x'] > .5
    pc['density'][cells] = 0.125
    pc['pressure'][cells] = 0.1

    # zero out velocities and particle type
    pc['velocity-x'][:] = 0.0
    pc['velocity-y'][:] = 0.0
    pc['tag'][:] = phd.ParticleTAGS.Real
    pc['type'][:] = phd.ParticleTAGS.Undefined

    return pc

# create inital state of the simulation
pc = create_particles(1.4)

domain = phd.DomainLimits(dim=2, xmin=0., xmax=1.)           # spatial size of problem 
boundary = phd.Boundary(domain,                              # reflective boundary condition
        boundary_type=phd.BoundaryType.Reflective)
mesh = phd.Mesh(boundary)                                    # tesselation algorithm
reconstruction = phd.PieceWiseConstant()                     # constant reconstruction
riemann = phd.HLL(reconstruction, gamma=1.4)                 # riemann solver
integrator = phd.MovingMesh(pc, mesh, riemann, regularize=1) # integrator 
solver = phd.Solver(integrator,                              # simulation driver
        cfl=0.5, tf=0.15, pfreq=1,
        relax_num_iterations=0,
        output_relax=False,
        fname='sod_2d_cartesian')
solver.solve()
