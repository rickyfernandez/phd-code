import phd
import numpy as np

a = 0.5
c = 0.25
e = c/a

G = 1.0
m1 = 1.0
m2 = 2.0
q = m1/m2 
m = m1 + m2
T = np.sqrt(4.*np.pi**2*a**3/(G*m))
dt = T/1000.

r0 = (1. - e)/(1. + q)*a
v0 = 1./(1. + q)*np.sqrt((1+e)/(1-e))*np.sqrt(G*m/a)

num_part = 2
particles = phd.HydroParticleCreator(num_part, dim=2)

particles["mass"][:] = np.array([m1, m2])
particles["position-x"][:] = np.array([r0, -q*r0])
particles["position-y"][:] = np.array([0., 0.])
particles["velocity-x"][:] = np.array([0., 0.])
particles["velocity-y"][:] = np.array([v0, -q*v0])

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[-1., -1.], xmax=[1., 1.],
        initial_radius=0.1) 

# setup gravity
gravity_tree = phd.GravityTree(barnes_angle=0.4,
        smoothing_length=0.0, calculate_potential=1)

# computation
integrator = phd.Nbody(dt=dt)
integrator.set_particles(particles)
integrator.set_domain_manager(domain_manager)
integrator.set_gravity_tree(gravity_tree)

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=10*T))

# output first step
output = phd.InitialOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# output every time step
output = phd.IterationInterval(iteration_interval=10)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name="two_body")
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
