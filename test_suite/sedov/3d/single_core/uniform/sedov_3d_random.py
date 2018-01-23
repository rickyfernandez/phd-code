import phd
import numpy as np

def create_particles(gamma=1.4):

    Lx = 1.       # domain size in x
    nx = 50       # particles per dim
    n = nx*nx*nx  # number of points

    # create particle container
    particles = phd.HydroParticleCreator(n, dim=3)
    part = 0
    np.random.seed(0)
    for i in range(nx):
        for j in range(nx):
            for k in range(nx):
                particles["position-x"][part] = np.random.rand()
                particles["position-y"][part] = np.random.rand()
                particles["position-z"][part] = np.random.rand()
                particles["ids"][part] = part
                part += 1

    # set ambient values
    particles["density"][:]  = 1.0     # density
    particles["pressure"][:] = 1.0E-5*(gamma-1)  # total energy

    # put all enegery in center particle
    r = 0.1
    cells = ( (particles["position-x"]-.5)**2\
            + (particles["position-y"]-.5)**2\
            + (particles["position-z"]-.5)**2 ) <= r**2
    particles["pressure"][cells] = 1.0/(4.0*np.pi*r**3/3.)*(gamma-1)

    # zero out the velocities and set particle type
    particles["velocity-x"][:] = 0.0
    particles["velocity-y"][:] = 0.0
    particles["velocity-z"][:] = 0.0
    particles["tag"][:] = phd.ParticleTAGS.Real
    particles["type"][:] = phd.ParticleTAGS.Undefined

    return particles

# unit square domain
minx = np.array([0., 0., 0.])
maxx = np.array([1., 1., 1.])
domain = phd.DomainLimits(minx, maxx, dim=3)

# computation related to boundaries
domain_manager = phd.DomainManager(initial_radius=0.4,
        search_radius_factor=1.25)

# create voronoi mesh
mesh = phd.Mesh(regularize=True, relax_iterations=10)

 #computation
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_domain_limits(domain)
integrator.set_particles(create_particles())
integrator.set_equation_state(phd.IdealGas())
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Reflective())
integrator.set_reconstruction(phd.PieceWiseConstant())

# add computation
integrator.set_riemann(phd.HLLC(boost=True))

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=0.1))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name="sedov", colored_logs=True)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
