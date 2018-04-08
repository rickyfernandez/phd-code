import phd
import numpy as np

# to run:
# $ mpirun -n 4 python gresho_2d_random.py
# for parallel or
# $ python gresho_2d_random.py
# for single core

def create_particles(dim=2, nx=45, Lx=1., gamma=1.4):

    dx = Lx/nx # spacing between particles
    n = nx*nx  # number of points

    # create particle container
    particles = phd.HydroParticleCreator(n, dim=2)
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
    particles["density"][:] = 1.0

    return particles

dim = 2; gamma = 1.4
particles = phd.distribute_initial_particles(
        create_particles, dim=dim, gamma=gamma)

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0.], xmax=[1., 1.],
        initial_radius=0.1)

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

sim_name = "gresho"
if phd._in_parallel:
    integrator.set_load_balance(phd.LoadBalance())
    sim_name = "mpi_gresho"

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=3.0))

# output initial state
output = phd.InitialOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# output every 0.5 time interval
output = phd.TimeInterval(time_interval=0.5)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name=sim_name)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
