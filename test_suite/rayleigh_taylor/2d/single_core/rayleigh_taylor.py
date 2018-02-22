import phd
import numpy as np

def create_particles(gamma=1.4):

    Lx = 1.
    nx = 64
    dx = Lx/nx

    Ly = 3.
    ny = 192
    dy = Ly/ny

    n = nx*ny

    # parameters of the problem
    center = 1.5

    p0 = 10.0
    grav = -1.0
    rho_1 = 1.0; rho_2 = 2.0
    amp = 1.0;   sigma = 0.1

    # create particle container
    particles = phd.HydroParticleCreator(n)
    part = 0
    for i in range(nx):
        for j in range(ny):

            x = (i+0.5)*dx
            y = (j+0.5)*dy

            if y < center:

                particles["density"][part] = rho_1
                particles["pressure"][part] = p0 + rho_1*grav*y

            else:

                particles["density"][part] = rho_2
                particles["pressure"][part] = p0 + rho_1*grav*center + rho_2*grav*(y-center)

            particles["position-x"][part] = x
            particles["position-y"][part] = y
            particles["velocity-y"][part] = amp*np.cos(2.*np.pi*x/1.)*np.exp(-(y-center)**2/sigma)
            particles["ids"][part] = part
            part += 1

    # zero out velocities and set particle type
    particles["velocity-x"][:] = 0.0
    particles["tag"][:] = phd.ParticleTAGS.Real
    particles["type"][:] = phd.ParticleTAGS.Undefined

    return particles

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0.], xmax=[1., 3.], initial_radius=0.1,
        search_radius_factor=2)

# create voronoi mesh
mesh = phd.Mesh(regularize=False, relax_iterations=0)

# computation
#integrator = phd.MovingMeshMUSCLHancock()
integrator = phd.StaticMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_riemann(phd.HLLC(boost=True))
integrator.set_particles(create_particles())
integrator.set_equation_state(phd.IdealGas())
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Reflective())
integrator.set_reconstruction(phd.PieceWiseLinear(limiter=0))

# source term
integrator.add_source_term(phd.ConstantGravity())

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=2.0))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.IterationInterval(iteration_interval=100)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name="rt", colored_logs=True)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
