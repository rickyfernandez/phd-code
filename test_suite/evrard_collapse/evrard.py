import phd
import numpy as np
import matplotlib.pyplot as plt

def create_particles(dim=3, gamma=1.4):

    L = 2.5     # domain size in x
    n = 33      # particles per dim
    dx = L/n
    num = n**3  # number of points

    M = 1.0
    R = 1.0
    G = 1.0
    u = 0.05*G*M/R
    c = 1.25

    rho_fac = M/(2.*np.pi*R**2)
    pre_fac = 2.*rho_fac*u/3.

    # create particle container
    particles = phd.HydroParticleCreator(num, dim=3)
    part = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):

                x = (i+0.5)*dx
                y = (j+0.5)*dx
                z = (k+0.5)*dx

                r = np.sqrt((x-c)**2 + (y-c)**2 + (z-c)**2)

                if r <= R:

                    # stretch
                    rn = r**1.5 + 0.001
                    xn = x-c; yn = y-c; zn = z - c

                    theta = np.arctan2(np.sqrt(xn**2 + yn**2),zn)
                    phi = np.arctan2(yn,xn)

                    particles["position-x"][part] = rn*np.sin(theta)*np.cos(phi) + c
                    particles["position-y"][part] = rn*np.sin(theta)*np.sin(phi) + c
                    particles["position-z"][part] = rn*np.cos(theta) + c

                    particles["density"][part] = rho_fac/rn
                    particles["pressure"][part] = pre_fac/rn

                else:
                    particles["density"][part] = 1.e-5*rho_fac/R
                    particles["pressure"][part] = 1.e-5*pre_fac/R

                    particles["position-x"][part] = x
                    particles["position-y"][part] = y 
                    particles["position-z"][part] = z

                particles["ids"][part] = part
                part += 1

    # zero out velocities and set particle type
    particles["velocity-x"][:] = 0.0
    particles["velocity-y"][:] = 0.0
    particles["velocity-z"][:] = 0.0

    return particles

dim = 3; gamma = 5./3.
particles = phd.distribute_initial_particles(
        create_particles, dim=dim, gamma=gamma)

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0., 0.], xmax=[2.5, 2.5, 2.5],
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

sim_name = "evrard"
if phd._in_parallel:
    integrator.set_load_balance(phd.LoadBalance())
    sim_name = "mpi_evrard"

gravity = phd.SelfGravity()
gravity_tree = phd.GravityTree(barnes_angle=0.5, smoothing_length=0.003, calculate_potential=1)
gravity_tree.register_fields(integrator.particles)
gravity_tree.add_fields(integrator.particles)
gravity_tree.set_domain_manager(domain_manager)
gravity_tree.initialize()
gravity.set_gravity(gravity_tree)
integrator.add_source_term(gravity)

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=0.81))
#simulation_time_manager.add_finish(phd.Time(time_max=3.0))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.TimeInterval(time_interval=0.01)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name=sim_name)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
