import phd
import numpy as np
from mpi4py import MPI

#example to run:
#$ mpirun -n 5 python sedov_2d_cartesian.py

if phd._rank == 0:

    gamma = 1.4

    Lx = 1.    # domain size in x
    nx = 101   # particles per dim
    n = nx*nx  # number of particles

    # create particle container
    particles_root = phd.HydroParticleCreator(n)
    part = 0
    np.random.seed(0)
    for i in range(nx):
        for j in range(nx):
            particles_root["position-x"][part] = np.random.rand()
            particles_root["position-y"][part] = np.random.rand()
            particles_root["ids"][part] = part
            part += 1

    # set ambient values
    particles_root["density"][:]  = 1.0               # density
    particles_root["pressure"][:] = 1.0E-5*(gamma-1)  # total energy

    # put all enegery in center particle
    r = 0.1
    cells = ((particles_root["position-x"]-.5)**2 + (particles_root["position-y"]-.5)**2) <= r**2
    particles_root["pressure"][cells] = 1.0/(np.pi*r**2)*(gamma-1)

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
            "pressure": None,
            "ids": None
            }

# tell each processor how many particles it will hold
send = phd._comm.scatter(send, root=0)

# allocate local particle container
particles = phd.HydroParticleCreator(send)

# import particles from root
fields = ["position-x", "position-y", "density", "pressure", "ids"]
for field in fields:
    phd._comm.Scatterv([particles_root[field], (lengths, disp)], particles[field])

particles["velocity-x"][:] = 0.0
particles["velocity-y"][:] = 0.0
particles["momentum-x"][:] = 0.0
particles["momentum-y"][:] = 0.0
particles["tag"][:] = phd.ParticleTAGS.Real
particles["type"][:] = phd.ParticleTAGS.Undefined

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0.], xmax=[1., 1.],
        initial_radius=0.1, search_radius_factor=2)

# create voronoi mesh
mesh = phd.Mesh(regularize=True, relax_iterations=8, max_iterations=10)

# computation
integrator = phd.MovingMeshMUSCLHancock()
integrator.set_mesh(mesh)
integrator.set_particles(particles)
integrator.set_riemann(phd.HLLC(boost=True))
integrator.set_equation_state(phd.IdealGas())
integrator.set_domain_manager(domain_manager)
integrator.set_load_balance(phd.LoadBalance())
integrator.set_boundary_condition(phd.Reflective())
integrator.set_reconstruction(phd.PieceWiseConstant())

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=0.1))

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.IterationInterval(iteration_interval=1)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name="sedov", colored_logs=True)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
