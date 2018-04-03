import phd
import numpy as np

def create_particles(gamma, nx=10):

    Lx = 1.    # domain size in x
    dx = Lx/nx # spacing between particles
    n = nx*nx  # number of points

    rho0 = 1.0
    vel0 = 0.0
    pre0 = 1.0

    A = 1.0e-6
    w = 2*np.pi
    k = 2*np.pi

    # create particle container
    particles = phd.HydroParticleCreator(n)

    particles["density"][:] = rho0
    particles["velocity-x"][:] = vel0
    particles["velocity-y"][:] = 0.0
    particles["pressure"][:] = pre0/gamma

    part = 0
    for i in range(nx):
        for j in range(nx):
            x = (i+0.5)*dx; y = (j+0.5)*dx

            particles["density"][part] += A*np.sin(k*x)
            particles["velocity-x"][part] += (w/k)*A/rho0*np.sin(k*x)
            particles["pressure"][part] += (w/k)**2*A*np.sin(k*x)

            particles["position-x"][part] = x
            particles["position-y"][part] = y
            particles["ids"][part] = part
            part += 1

    return particles


for i in [10, 20, 40, 80, 160]:

    gamma = 5./3.
    particles = create_particles(gamma, nx=i)

    # computation related to boundaries
    domain_manager = phd.DomainManager(
            xmin=[0., 0.], xmax=[1., 1.], initial_radius=0.1)

    # create voronoi mesh
    mesh = phd.Mesh(regularize=False)

    # computation
    integrator = phd.MovingMeshMUSCLHancock()
    integrator.set_mesh(mesh)
    integrator.set_riemann(phd.HLLC())
    integrator.set_particles(particles)
    integrator.set_domain_manager(domain_manager)
    integrator.set_boundary_condition(phd.Periodic())
    integrator.set_reconstruction(phd.PieceWiseLinear(gizmo_limiter=False))
    integrator.set_equation_state(phd.IdealGas(gamma=gamma))

    # add finish criteria
    simulation_time_manager = phd.SimulationTimeManager()
    simulation_time_manager.add_finish(phd.Time(time_max=1.0))

    # output last step
    output = phd.FinalOutput()
    output.set_writer(phd.Hdf5())
    simulation_time_manager.add_output(output)

    # output initial data 
    output = phd.InitialOutput()
    output.set_writer(phd.Hdf5())
    simulation_time_manager.add_output(output)

    # Create simulator
    simulation = phd.Simulation(simulation_name="linear_wave_"+str(i), colored_logs=True)
    simulation.set_integrator(integrator)
    simulation.set_simulation_time_manager(simulation_time_manager)
    simulation.initialize()
    simulation.solve()
