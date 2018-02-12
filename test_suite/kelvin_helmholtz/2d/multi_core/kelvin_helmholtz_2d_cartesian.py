import phd
import numpy as np

#example to run:
#$ mpirun -n 5 python sedov_2d_cartesian.py

if phd._rank == 0:

    gamma = 1.4

    Lx = 1.    # domain size in x
    nx = 100   # particles per dim
    n = nx*nx  # number of points

    rho_1 = 1.0; rho_2 = 2.0
    vel = 0.5; amp = 0.05

    dx = Lx/nx # spacing between particles

    # create particle container
    particles_root = phd.HydroParticleCreator(n)
    part = 0
    for i in range(nx):
        for j in range(nx):

            x = (i+0.5)*dx
            y = (j+0.5)*dx

            pert = amp*np.sin(4.*np.pi*x)

            if 0.25 < y and y < 0.75:

                particles_root["density"][part] = rho_1
                particles_root["velocity-x"][part] = -(vel + pert)

            else:

                particles_root["density"][part] = rho_2
                particles_root["velocity-x"][part] = vel + pert


            particles_root["position-x"][part] = x
            particles_root["position-y"][part] = y
            particles_root["ids"][part] = part
            part += 1

    # how many particles to each process
    nsect, extra = divmod(n, phd._size)
    lengths = extra*[nsect+1] + (phd._size-extra)*[nsect]
    send = np.array(lengths)

    # how many particles 
    disp = np.zeros(phd._size, dtype=np.int32)
    for i in range(1,phd._size):
        disp[i] = send[i-1] + disp[i-1]

else:

    lengths = disp = send = None
    particles_root = {
            "position-x": None,
            "position-y": None,
            "velocity-x": None,
            "density": None,
            "ids": None
            }

# tell each processor how many particles it will hold
send = phd._comm.scatter(send, root=0)

# allocate local particle container
particles = phd.HydroParticleCreator(send)

# import particles from root
fields = ["position-x", "position-y", "velocity-x", "density", "ids"]
for field in fields:
    phd._comm.Scatterv([particles_root[field], (lengths, disp)], particles[field])

particles["pressure"][:] = 2.5
particles["velocity-y"][:] = 0.0
particles["tag"][:] = phd.ParticleTAGS.Real
particles["type"][:] = phd.ParticleTAGS.Undefined

# unit square domain
minx = np.array([0., 0.])
maxx = np.array([1., 1.])
domain = phd.DomainLimits(minx, maxx)

# computation related to boundaries
domain_manager = phd.DomainManager(initial_radius=0.1,
        search_radius_factor=2)

# create voronoi mesh
mesh = phd.Mesh(regularize=True, relax_iterations=0)

# computation
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_riemann(phd.HLLC(boost=True))
integrator.set_domain_limits(domain)
integrator.set_particles(particles)
integrator.set_equation_state(phd.IdealGas())
integrator.set_domain_manager(domain_manager)
integrator.set_load_balance(phd.LoadBalance())
integrator.set_boundary_condition(phd.Periodic())
integrator.set_reconstruction(phd.PieceWiseLinear())

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=2.5))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.IterationInterval(iteration_interval=25)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name="kh", colored_logs=True)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
