import phd
import numpy as np

def create_particles(gamma=1.4):

    Lx = 1.    # domain size in x
    nx = 101   # particles per dim
    n = nx*nx  # number of points

    dx = Lx/nx # spacing between particles

    # create particle container
    pc = phd.HydroParticleCreator(n)
    part = 0
    for i in range(nx):
        for j in range(nx):
            pc['position-x'][part] = (i+0.5)*dx
            pc['position-y'][part] = (j+0.5)*dx
            pc['ids'][part] = part
            part += 1

    # set ambient values
    pc['density'][:]  = 1.0               # density
    pc['pressure'][:] = 1.0E-5*(gamma-1)  # total energy

    # put all enegery in center particle
    r = dx * .51
    cells = ((pc['position-x']-.5)**2 + (pc['position-y']-.5)**2) <= r**2
    pc['pressure'][cells] = 1.0/(dx*dx)*(gamma-1)

    # zero out velocities and set particle type
    pc['velocity-x'][:] = 0.0
    pc['velocity-y'][:] = 0.0
    pc['tag'][:] = phd.ParticleTAGS.Real
    pc['type'][:] = phd.ParticleTAGS.Undefined

    return pc

# unit square domain
minx = np.array([0., 0.])
maxx = np.array([1., 1.])
domain = phd.DomainLimits(minx, maxx)

# computation related to boundaries
domain_manager = phd.DomainManager(param_initial_radius=0.1,
        param_search_radius_factor=1.25)

# computation
integrator = phd.MovingMeshMUSCLHancock()

# add mesh related
integrator.set_domain_limits(domain)
integrator.set_domain_manager(domain_manager)
integrator.set_boundary_condition(phd.Reflective())

mesh = phd.Mesh(regularize=False)
integrator.set_mesh(mesh)

# add particle information
particles = create_particles()
integrator.set_particles(particles)
integrator.set_equation_state(phd.IdealGas())

# add computation
integrator.set_riemann(phd.HLLC(param_boost=False))
#integrator.set_reconstruction(phd.PieceWiseConstant())
integrator.set_reconstruction(phd.PieceWiseLinear())

# Create simulator
simulation = phd.Simulation(simulation_name="constant_run", colored_logs=True)
simulation.set_integrator(integrator)

# add outputs
simulation_time_manager = phd.SimulationTimeManager()
#simulation_time_manager.add_finish(phd.Iteration(iteration_max=1))  # finish simulation after one step
simulation_time_manager.add_finish(phd.Time(time_max=0.1))  # finish simulation after one step

output = phd.IterationInterval(iteration_interval=1)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.InitialOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# run simulation
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
#integrator.initialize()
#print 'particles', particles.properties.keys()
#print 'faces', mesh.faces.properties.keys()
