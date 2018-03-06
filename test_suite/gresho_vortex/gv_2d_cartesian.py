import phd
import numpy as np

def create_particles(gamma=1.4):

    Lx = 1.    # domain size in x
    nx = 64   # particles per dim
    n = nx*nx  # number of points

    dx = Lx/nx # spacing between particles

    # create particle container
    particles = phd.HydroParticleCreator(n)
    part = 0
    for i in range(nx):
        for j in range(nx):

            x = (i+0.5)*dx - 0.5
            y = (j+0.5)*dx - 0.5

            theta = np.arctan2(y, x)
            r = np.sqrt(x**2 + y**2)

            if 0 <= r < 0.2:
                vtheta = 5*r
                press = 5 + 25./2*r**2

            elif 0.2 <= r < 0.4:
                vtheta = 2 - 5*r
                press = 9 + 25./2*r**2 - 20.*r + 4*np.log(r/0.2)

            else:
                vtheta = 0.
                press = 3 + 4*np.log(2)

            particles["position-x"][part] = x + 0.5
            particles["position-y"][part] = y + 0.5
            particles["velocity-x"][part] = -np.sin(theta)*vtheta
            particles["velocity-y"][part] =  np.cos(theta)*vtheta
            particles["pressure"][part] = press 
            particles["ids"][part] = part
            part += 1

    # set ambient values
    particles["density"][:] = 1.0  # density
    particles["tag"][:] = phd.ParticleTAGS.Real
    particles["type"][:] = phd.ParticleTAGS.Undefined

    return particles

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0.], xmax=[1., 1.], initial_radius=0.1,
        search_radius_factor=2)

# create voronoi mesh
mesh = phd.Mesh(regularize=True, relax_iterations=0)

# computation
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_riemann(phd.HLLC(boost=True))
integrator.set_particles(create_particles())
integrator.set_equation_state(phd.IdealGas())
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Periodic())
integrator.set_reconstruction(phd.PieceWiseLinear(limiter=0))

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=3.0))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.IterationInterval(iteration_interval=100)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name="gv", colored_logs=True)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
