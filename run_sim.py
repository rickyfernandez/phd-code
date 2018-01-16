import phd
import numpy as np

def create_particles(gamma=1.4):
    """Create initial state of primitive particles
    in sedov problem.

    Particles are placed in an uniform grid. All the energy is
    deposited at the center particle.

    Parameters
    ---------
    gamma : float
        Ratio of specific heats.

    """

    Lx = 1.    # domain size in x
    nx = 101   # particles per dim
    n = nx*nx  # number of particles

    dx = Lx/nx # spacing between particles

    # create particle container
    particles = phd.HydroParticleCreator(n)
    part = 0
    for i in range(nx):
        for j in range(nx):
            particles["position-x"][part] = (i+0.5)*dx
            particles["position-y"][part] = (j+0.5)*dx
            particles["ids"][part] = part
            part += 1

    # set ambient values
    particles["density"][:]  = 1.0               # density
    particles["pressure"][:] = 1.0E-5*(gamma-1)  # total energy

    # put all enegery in center particle
    r = dx * .51
    cells = ((particles["position-x"]-.5)**2 + (particles["position-y"]-.5)**2) <= r**2
    particles["pressure"][cells] = 1.0/(dx*dx)*(gamma-1)

    # zero out velocities and set particle type
    particles["velocity-x"][:] = 0.0
    particles["velocity-y"][:] = 0.0
    particles["tag"][:] = phd.ParticleTAGS.Real
    particles["type"][:] = phd.ParticleTAGS.Undefined

    return particles

# This is serial test problem of the sedov blast wave problem. The 
# problem is setup with uniform density and zero velocity. The 
# center most particle is given a large amount of energy. In this
# example the equations are solved by the MUSCL Hancock method with
# linear reconstruction and the HLLC solver. It is found regularizing
# the mesh generator motions leads to failure so regularization is
# not added.

# create initial sedov particles
particles = create_particles()

# create square domain
minx = np.array([0., 0.])
maxx = np.array([1., 1.])
domain = phd.DomainLimits(minx, maxx)

# computation related to boundaries
domain_manager = phd.DomainManager(param_initial_radius=0.1,
        param_search_radius_factor=1.25)

# create integrator to solve the equations 
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_domain_limits(domain)                 # domain of the simulation    
integrator.set_riemann(phd.HLLC(boost=True))         # riemann solver
integrator.set_particles(create_particles())         # create initial particles
integrator.set_equation_state(phd.IdealGas())        # equation of state
integrator.set_domain_manager(domain_manager)        # domain manager
integrator.set_mesh(phd.Mesh(regularize=False)       # voronoi mesh 
integrator.set_boundary_condition(phd.Reflective())  # reflective boundary condition
integrator.set_reconstruction(phd.PieceWiseLinear()) # primitive reconstruction

# add signal to finish the simulation
simulation_time_manager = phd.SimulationTimeManager()      # time manager
simulation_time_manager.add_finish(phd.Time(time_max=0.1)) # max simulation time

# add data outputters
output = phd.FinalOutput()                  # output final time
output.set_writer(phd.Hdf5())               # write data in hdf5 format
simulation_time_manager.add_output(output)

# Create simulation
simulation = phd.Simulation(simulation_name="sedov", colored_logs=True)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.set_integrator(integrator)

# run simulation
simulation.initialize()
simulation.solve()
