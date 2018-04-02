import phd
import numpy as np

# to run:
# $ mpirun -n 4 python sedov_2d_cartesian.py
# for parallel or
# $ python sedov_2d_cartesian.py
# for single core

def create_particles(dim=2, nx=100, Lx=1.0, diaphragm=0.5, gamma=1.4):

    dx = Lx/nx # spacing between particles
    n = nx*nx  # number of particles

    # create particle container
    particles = phd.HydroParticleCreator(n, dim=2)
    part = 0
    np.random.seed(0)
    for i in range(nx):
        for j in range(nx):
            particles_root["position-x"][part] = np.random.rand()
            particles_root["position-y"][part] = np.random.rand()
            particles["ids"][part] = part
            part += 1

    # set ambient values
    particles["density"][:]  = 1.0
    particles["pressure"][:] = 1.0
    particles["velocity-x"][:] = 0.0
    particles["velocity-y"][:] = 0.0

    cells = particles["position-x"] > diaphragm
    particles["density"][cells] = 0.125
    particles["pressure"][cells] = 0.1

    return particles

particles = phd.distribute_initial_particles(create_particles, dim=2)

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0.], xmax=[1., 1.],
        initial_radius=0.1, search_radius_factor=2)

# create voronoi mesh
mesh = phd.Mesh(regularize=False)

# computation
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_riemann(phd.HLLC())
integrator.set_particles(particles)
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Reflective())
integrator.set_reconstruction(phd.PieceWiseLinear())
integrator.set_equation_state(phd.IdealGas(gamma=1.4))

sim_name = "sod"
if phd._in_parallel:
    integrator.set_load_balance(phd.LoadBalance())
    sim_name = "mpi_sod"

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
