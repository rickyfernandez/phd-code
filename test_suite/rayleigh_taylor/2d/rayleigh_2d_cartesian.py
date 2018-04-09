import phd
import numpy as np

def create_particles(dim=2, gamma=1.4):

    nx = 50
    ny = 150
    Lx = 1.
    Ly = 3.

    if phd._in_parallel:
        # since we don't have rectangular
        # boundaries in parallel
        nx = 150
        ny = 150
        Lx = 3.
        Ly = 3.

    dx = Lx/nx
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
            particles["velocity-y"][part] = amp*np.cos(2.*np.pi*x/1.)*np.exp(-(y-center)**2/sigma**2)
            particles["ids"][part] = part
            part += 1

    # zero out velocities and set particle type
    particles["velocity-x"][:] = 0.0

    return particles

dim = 2; gamma = 1.4
particles = phd.distribute_initial_particles(
        create_particles, dim=dim, gamma=gamma)

# computation related to boundaries
if phd._in_parallel:
    domain_manager = phd.DomainManager(
            xmin=[0., 0.], xmax=[3., 3.], initial_radius=0.3)
else:
    domain_manager = phd.DomainManager(
            xmin=[0., 0.], xmax=[1., 3.], initial_radius=0.1)

# create voronoi mesh
mesh = phd.Mesh()

# computation
integrator = phd.MovingMeshMUSCLHancock()
integrator = phd.StaticMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_riemann(phd.HLLC())
integrator.set_particles(particles)
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Reflective())
integrator.set_reconstruction(phd.PieceWiseLinear())
integrator.set_equation_state(phd.IdealGas(gamma=gamma))

sim_name = "rayleigh_moving"
if phd._in_parallel:
    integrator.set_load_balance(phd.LoadBalance())
    sim_name = "mpi_rayleigh"

# source term
integrator.add_source_term(phd.ConstantGravity())

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=2.0))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.TimeInterval(time_interval=0.5)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name=sim_name)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
