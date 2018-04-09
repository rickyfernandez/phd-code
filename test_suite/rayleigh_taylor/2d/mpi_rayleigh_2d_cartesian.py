import phd
import numpy as np

#to run:
#$ mpirun -n 4 python sedov_2d_cartesian.py

if phd._rank == 0:

    Lx = 3.
    nx = 144
    dx = Lx/nx

    Ly = 3.
    ny = 144
    dy = Ly/ny

    n = nx*ny

    # parameters of the problem
    center = 1.5

    p0 = 10.0
    grav = -1.0
    rho_1 = 1.0; rho_2 = 2.0
    amp = 1.0;   sigma = 0.1

    # create particle container
    particles_root = phd.HydroParticleCreator(n)
    part = 0
    for i in range(nx):
        for j in range(ny):

            x = (i+0.5)*dx
            y = (j+0.5)*dy

            if y < center:

                particles_root["density"][part] = rho_1
                particles_root["pressure"][part] = p0 + rho_1*grav*y

            else:

                particles_root["density"][part] = rho_2
                particles_root["pressure"][part] = p0 + rho_1*grav*center + rho_2*grav*(y-center)

            particles_root["position-x"][part] = x
            particles_root["position-y"][part] = y
            particles_root["velocity-y"][part] = amp*np.cos(2.*np.pi*x/1.)*np.exp(-(y-center)**2/sigma)
            particles_root["ids"][part] = part
            part += 1

    # how many particles to each process
    nsect, extra = divmod(n, phd._size)
    lengths = extra*[nsect+1] + (phd._size-extra)*[nsect]
    send = np.array(lengths)

    # how many particles 
    disp = np.zeros(phd._size, dtype=np.int32)
    for i in range(1, phd._size):
        disp[i] = send[i-1] + disp[i-1]

else:

    lengths = disp = send = None
    particles_root = {
            "position-x": None,
            "position-y": None,
            "density": None,
            "velocity-y": None,
            "pressure": None,
            "ids": None
            }

# tell each processor how many particles it will hold
send = phd._comm.scatter(send, root=0)

# allocate local particle container
particles = phd.HydroParticleCreator(send)

# import particles from root
fields = ["position-x", "position-y", "density", "velocity-y", "pressure", "ids"]
for field in fields:
    phd._comm.Scatterv([particles_root[field], (lengths, disp)], particles[field])

particles["velocity-x"][:] = 0.0
particles["tag"][:] = phd.ParticleTAGS.Real
particles["type"][:] = phd.ParticleTAGS.Undefined

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0.], xmax=[3., 3.], initial_radius=0.1,
        search_radius_factor=2)

# create voronoi mesh
mesh = phd.Mesh(regularize=True, relax_iterations=0)

# computation
integrator = phd.MovingMeshMUSCLHancock()
#integrator = phd.StaticMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_riemann(phd.HLLC(boost=True))
integrator.set_particles(particles)
integrator.set_equation_state(phd.IdealGas())
integrator.set_domain_manager(domain_manager)
integrator.set_load_balance(phd.LoadBalance())
integrator.set_boundary_condition(phd.Reflective())
integrator.set_reconstruction(phd.PieceWiseLinear(limiter="arepo", gizmo_limiter=True))

# source term
integrator.add_source_term(phd.ConstantGravity())

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=2.0))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.IterationInterval(iteration_interval=100)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name="rt", colored_logs=True)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
