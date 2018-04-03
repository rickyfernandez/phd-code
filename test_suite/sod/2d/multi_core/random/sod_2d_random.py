import phd
import numpy as np

# to run:
# $ mpirun -n 4 python sedov_2d_cartesian.py
# for parallel or
# $ python sedov_2d_cartesian.py
# for single core

def create_particles(dim=2, n=10000, diaphragm=0.5, gamma=1.4):

    # create particle container
    particles = phd.HydroParticleCreator(n, dim=2)
    np.random.seed(0)
    for i in range(n):
        particles["position-x"][i] = np.random.rand()
        particles["position-y"][i] = np.random.rand()
        particles["ids"][i] = i

    # set ambient values
    particles["density"][:]  = 1.0
    particles["pressure"][:] = 1.0
    particles["velocity-x"][:] = 0.0
    particles["velocity-y"][:] = 0.0

    cells = particles["position-x"] > diaphragm
    particles["density"][cells] = 0.125
    particles["pressure"][cells] = 0.1

    return particles

dim = 2; gamma = 1.4
particles = phd.distribute_initial_particles(
        create_particles, dim=dim, gamma=gamma)

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0.], xmax=[1., 1.],
        initial_radius=0.1, search_radius_factor=2)

# create voronoi mesh
mesh = phd.Mesh(relax_iterations=8, max_iterations=10)

# computation
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_particles(particles)
integrator.set_riemann(phd.HLLC())
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Reflective())
integrator.set_reconstruction(phd.PieceWiseLinear())
integrator.set_equation_state(phd.IdealGas(gamma=gamma))

sim_name = "sod"
if phd._in_parallel:
    integrator.set_load_balance(phd.LoadBalance())
    sim_name = "sod_mpi"

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=0.15))

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
