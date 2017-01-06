import phd
import numpy as np

def create_particles(gamma):

    Lx = 1.    # domain size in x
    nx = 101   # particles per dim
    n = nx*nx  # number of points

    # create particle container
    pc = phd.HydroParticleCreator(n)
    part = 0
    np.random.seed(0)
    for i in range(nx):
        for j in range(nx):
            pc['position-x'][part] = np.random.rand()
            pc['position-y'][part] = np.random.rand()
            pc['ids'][part] = part
            part += 1

    # set ambient values
    pc['density'][:]  = 1.0     # density
    pc['pressure'][:] = 1.0E-5*(gamma-1)  # total energy

    # put all enegery in center particle
    r = 0.1
    cells = ((pc['position-x']-.5)**2 + (pc['position-y']-.5)**2) <= r**2
    pc['pressure'][cells] = 1.0/(np.pi*r**2)*(gamma-1)

    # zero out the velocities and set particle type
    pc['velocity-x'][:] = 0.0
    pc['velocity-y'][:] = 0.0
    pc['tag'][:] = phd.ParticleTAGS.Real
    pc['type'][:] = phd.ParticleTAGS.Undefined

    return pc

# simulation driver
sim = phd.Simulation(
        cfl=0.5, tf=0.1, pfreq=1,
        relax_num_iterations=10,
        output_relax=False,
        fname='sedov_2d_uniform')

sim.add_particles(create_particles(1.4))                                  # create inital state of the simulation
sim.add_domain(phd.DomainLimits(dim=2, xmin=0., xmax=1.))                 # spatial size of problem 
sim.add_boundary(phd.Boundary(boundary_type=phd.BoundaryType.Reflective)) # reflective boundary condition
sim.add_mesh(phd.Mesh())                                                  # tesselation algorithm
sim.add_reconstruction(phd.PieceWiseLinear(limiter=1))                    # Linear reconstruction
#sim.add_reconstruction(phd.PieceWiseConstant(limiter=1))                    # Linear reconstruction
sim.add_riemann(phd.HLLC(gamma=1.4, boost=0))                             # riemann solver
sim.add_integrator(phd.MovingMesh(regularize=1))                          # Integrator

# run the simulation
sim.solve()
