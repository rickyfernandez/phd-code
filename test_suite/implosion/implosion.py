import phd
import numpy as np

def create_particles(gamma):

    Lx = 1.    # domain size in x
    nx = 128   # particles per dim
    n = nx*nx  # number of points

    dx = Lx/nx # spacing between particles

    # create particle container
    pc = phd.HydroParticleCreator(n)

    # set ambient values
    pc['density'][:]  = 1.0  # density
    pc['pressure'][:] = 1.0  # total energy

    part = 0
    for i in range(nx):
        for j in range(nx):

            pc['position-x'][part] = (i+0.5)*dx
            pc['position-y'][part] = (j+0.5)*dx
            pc['ids'][part] = part

            if pc['position-x'][part] + pc['position-y'][part] < 0.5:
                pc['density'][part] = 0.125
                pc['pressure'][part] = 0.14

            part += 1

    # zero out velocities and set particle type
    pc['velocity-x'][:] = 0.0
    pc['velocity-y'][:] = 0.0
    pc['tag'][:] = phd.ParticleTAGS.Real
    pc['type'][:] = phd.ParticleTAGS.Undefined

    return pc

# simulation driver
sim = phd.Simulation(
        cfl=0.5, tf=2.5, pfreq=25,
        relax_num_iterations=0,
        output_relax=False,
        fname='implosion')

sim.add_component(create_particles(1.4))                                   # create inital state of the simulation
sim.add_component(phd.DomainLimits(dim=2, xmin=0., xmax=1.))               # spatial size of problem 
sim.add_component(phd.Boundary(boundary_type=phd.BoundaryType.Reflective)) # reflective boundary condition
sim.add_component(phd.Mesh())                                              # tesselation algorithm
sim.add_component(phd.PieceWiseLinear())                                   # Linear reconstruction
#sim.add_component(phd.PieceWiseConstant())                                   # Linear reconstruction
sim.add_component(phd.HLLC(gamma=1.4))                                     # riemann solver
sim.add_component(phd.MovingMesh(regularize=1))                            # Integrator

# run the simulation
sim.solve()
