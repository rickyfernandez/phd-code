import phd
import numpy as np

# to run:
# $ mpirun -n 4 python explosion_2d_random.py
# for parallel or
# $ python explosion_2d_random.py
# for single core

def create_particles(dim=2, n=10000, gamma=1.4):

    # create particle container
    particles = phd.HydroParticleCreator(n, dim=2)

    c = 0.5
    for i in range(n):

        x = np.random.rand()
        y = np.random.rand()

        if (x-c)**2 + (y-c)**2 <= 0.25**2:
            particles["density"][i] = 1.0
            particles["pressure"][i] = 1.0
        else:
            particles["density"][i] = 0.125
            particles["pressure"][i] = 0.1

        particles["position-x"][i] = x
        particles["position-y"][i] = y
        particles["ids"][i] = i

    # zero out velocities and set particle type
    particles["velocity-x"][:] = 0.0
    particles["velocity-y"][:] = 0.0

    return particles

dim = 2; gamma = 5./3.
particles = phd.distribute_initial_particles(
        create_particles, dim=dim, gamma=gamma)

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0.], xmax=[1., 1.],
        initial_radius=0.1)

# create voronoi mesh
mesh = phd.Mesh(relax_iterations=10)

# computation
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_riemann(phd.HLLC())
integrator.set_particles(particles)
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Reflective())
integrator.set_reconstruction(phd.PieceWiseLinear())
integrator.set_equation_state(phd.IdealGas(gamma=gamma))

sim_name = "explosion"
if phd._in_parallel:
    integrator.set_load_balance(phd.LoadBalance())
    sim_name = "explosion_mpi"

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=0.1))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name=sim_name)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
