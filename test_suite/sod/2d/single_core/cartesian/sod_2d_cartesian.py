import phd
import numpy as np

def create_particles(gamma):

    Lx = 1.    # domain size in x
    nx = 100    # particles per dim
    n = nx*nx  # number of points

    dx = Lx/nx # spacing between particles

    # create particle container
    pc = phd.HydroParticleCreator(n)
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

# simulation driver
sim = phd.Simulation(
        cfl=0.5, tf=0.15, pfreq=1,
        relax_num_iterations=0,
        output_relax=False,
        fname='sod_2d_cartesian')

sim.add_particles(create_particles(1.4))                                  # create inital state of the simulation
sim.add_domain(phd.DomainLimits(dim=2, xmin=0., xmax=1.))                 # spatial size of problem 
sim.add_boundary(phd.Boundary(boundary_type=phd.BoundaryType.Reflective)) # reflective boundary condition
sim.add_mesh(phd.Mesh())                                                  # tesselation algorithm
sim.add_reconstruction(phd.PieceWiseLinear(limiter=0, boost=1))           # Linear reconstruction
sim.add_riemann(phd.HLLC(gamma=1.4))                                      # riemann solver
#sim.add_riemann(phd.Exact(gamma=1.4))                                      # riemann solver
sim.add_integrator(phd.MovingMesh(regularize=1))                          # Integrator

sim.solve()
