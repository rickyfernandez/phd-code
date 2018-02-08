import phd
import numpy as np

def create_particles(gamma=1.4):

    Lx = 1.    # domain size in x
    nx = 100   # particles per dim
    n = nx*nx  # number of points

    dx = Lx/nx # spacing between particles

    # create particle container
    particles = phd.HydroParticleCreator(n)
    part = 0
    for i in range(nx):
        for j in range(nx):
            particles["position-x"][part] = (i+0.5)*dx
            particles["position-y"][part] = (j+0.5)*dx
            particles["ids"][part] = part
            part += 1

    # set ambient values
    particles["density"][:]  = 1.0  # density
    particles["pressure"][:] = 1.0  # total energy

    cells = particles["position-x"] > .5
    particles["density"][cells] = 0.125
    particles["pressure"][cells] = 0.1

    # zero out velocities and set particle type
    particles["velocity-x"][:] = 0.0
    particles["velocity-y"][:] = 0.0
    particles["tag"][:] = phd.ParticleTAGS.Real
    particles["type"][:] = phd.ParticleTAGS.Undefined

    return particles

# unit square domain
minx = np.array([0., 0.])
maxx = np.array([1., 1.])
domain = phd.DomainLimits(minx, maxx)

# computation related to boundaries
domain_manager = phd.DomainManager(initial_radius=0.1,
        search_radius_factor=2)

# create voronoi mesh
mesh = phd.Mesh(regularize=False, relax_iterations=0)

# computation
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_domain_limits(domain)
integrator.set_riemann(phd.HLLC(boost=True))
integrator.set_particles(create_particles())
integrator.set_equation_state(phd.IdealGas())
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Reflective())
integrator.set_reconstruction(phd.PieceWiseLinear(limiter=0))

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=0.15))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.IterationInterval(iteration_interval=1)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name="sod", colored_logs=True)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
