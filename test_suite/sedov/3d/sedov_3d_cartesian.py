import phd
import numpy as np

# to run:
# $ mpirun -n 4 python sedov_3d_cartesian.py
# for parallel or
# $ python sedov_3d_cartesian.py
# for single core

def create_particles(dim=3, nx=45, Lx=1.0, gamma=1.4):

    dx = Lx/nx # spacing between particles
    n = nx**3  # number of points


    # create particle container
    particles = phd.HydroParticleCreator(n, dim=3)
    part = 0
    np.random.seed(0)
    for i in range(nx):
        for j in range(nx):
            for k in range(nx):
                particles["position-x"][part] = (i+0.5)*dx + 1.0e-8*dx*np.random.rand()
                particles["position-y"][part] = (j+0.5)*dx + 1.0e-8*dx*np.random.rand()
                particles["position-z"][part] = (k+0.5)*dx + 1.0e-8*dx*np.random.rand()
                particles["ids"][part] = part
                part += 1

    # set ambient values
    particles["density"][:]  = 1.0               # density
    particles["pressure"][:] = 1.0E-6*(gamma-1)  # total energy
    particles["velocity-x"][:] = 0.0
    particles["velocity-y"][:] = 0.0
    particles["velocity-z"][:] = 0.0

    # put all enegery in center particle
    r = dx * .51
    cell = ((particles["position-x"]-.5)**2 +\
             (particles["position-y"]-.5)**2 +\
             (particles["position-z"]-.5)**2) <= r**2

    particles["pressure"][cell] = 1.0/(dx*dx*dx)*(gamma-1)

    return particles

dim = 3; gamma = 5./3.
particles = phd.distribute_initial_particles(
        create_particles, dim=dim, gamma=gamma)

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0., 0.], xmax=[1., 1., 1.],
        initial_radius=0.1)

# create voronoi mesh
mesh = phd.Mesh()

# computation
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_riemann(phd.HLLC())
integrator.set_particles(particles)
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Reflective())
integrator.set_reconstruction(phd.PieceWiseLinear())
integrator.set_equation_state(phd.IdealGas(gamma=gamma))

sim_name = "sedov"
if phd._in_parallel:
    integrator.set_load_balance(phd.LoadBalance())
    sim_name = "mpi_sedov"

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=0.06))

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
