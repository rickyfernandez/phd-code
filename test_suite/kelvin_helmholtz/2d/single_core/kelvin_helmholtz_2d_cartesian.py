import phd
import numpy as np

# to run:
# $ mpirun -n 4 python gresho_2d_random.py
# for parallel or
# $ python gresho_2d_random.py
# for single core

def create_particles(dim, gamma):

    Lx = 1.    # domain size in x
    nx = 50   # particles per dim
    n = nx*nx  # number of points

    rho_1 = 1.0; rho_2 = 2.0
    vel = 0.5; amp = 0.05
    sigma=0.05/np.sqrt(2)

    dx = Lx/nx # spacing between particles

    # create particle container
    particles = phd.HydroParticleCreator(n, dim=2)
    part = 0
    for i in range(nx):
        for j in range(nx):

            x = (i+0.5)*dx
            y = (j+0.5)*dx

            pert = amp*np.sin(4.*np.pi*x)

            if 0.25 < y and y < 0.75:

                #particles["density"][part] = rho_1
                #particles["velocity-x"][part] = -(vel + pert)
                particles["density"][part] = rho_2
                particles["velocity-x"][part] = vel

            else:

                #particles["density"][part] = rho_2
                #particles["velocity-x"][part] = vel + pert
                particles["density"][part] = rho_1
                particles["velocity-x"][part] = -vel


            particles["velocity-y"][part] = 0.1*np.sin(4*np.pi*x)*(np.exp(-(y-0.25)**2/(2*sigma**2)) +\
                    np.exp(-(y-0.75)**2/(2*sigma**2)))
            particles["position-x"][part] = x
            particles["position-y"][part] = y
            particles["ids"][part] = part
            part += 1

    particles["pressure"][:] = 2.5
    #particles["velocity-y"][:] = 0.0

    return particles

dim = 2; gamma = 5./3.
particles = phd.distribute_initial_particles(
        create_particles, dim=dim, gamma=gamma)

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0.], xmax=[1., 1.], initial_radius=0.1)

# create voronoi mesh
mesh = phd.Mesh()

# computation
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_riemann(phd.HLLC())
integrator.set_particles(particles)
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Periodic())
integrator.set_reconstruction(phd.PieceWiseLinear())
integrator.set_equation_state(phd.IdealGas(gamma=gamma))

sim_name = "kelvin"
if phd._in_parallel:
    integrator.set_load_balance(phd.LoadBalance())
    sim_name = "mpi_kelvin"

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=2.0))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# output every 0.1 time interval
output = phd.TimeInterval(time_interval=0.1)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name=sim_name)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
