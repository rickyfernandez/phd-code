import phd
import numpy as np

def create_particles(gamma):

    Lx = 1.    # domain size in x
    nx = 100   # particles per dim
    n = nx*nx  # number of points

    rho_1 = 1.0; rho_2 = 2.0
    vel = 0.5; amp = 0.05

    dx = Lx/nx # spacing between particles

    # create particle container
    particles = phd.HydroParticleCreator(n)
    part = 0
    for i in range(nx):
        for j in range(nx):

            x = (i+0.5)*dx
            y = (j+0.5)*dx

            pert = amp*np.sin(4.*np.pi*x)

            if 0.25 < y and y < 0.75:

                particles["density"][part] = rho_1
                particles["velocity-x"][part] = -(vel + pert)

            else:

                particles["density"][part] = rho_2
                particles["velocity-x"][part] = vel + pert


            particles["position-x"][part] = x
            particles["position-y"][part] = y
            particles["ids"][part] = part
            part += 1

    particles["pressure"][:] = 2.5
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
mesh = phd.Mesh(regularize=True, relax_iterations=0)

# computation
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_riemann(phd.HLLC(boost=True))
integrator.set_domain_limits(domain)
integrator.set_particles(create_particles(1.4))
integrator.set_equation_state(phd.IdealGas())
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Periodic())
integrator.set_reconstruction(phd.PieceWiseLinear(limiter=0))

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=2.5))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.IterationInterval(iteration_interval=25)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name="kh", colored_logs=True)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
