import phd
import numpy as np

def make_plummer(particles, M=1., R=1., r_cut_off=30):

    N = particles.get_carray_size()

    E = 3./64.*np.pi*M*M/R
    np.random.seed(0)
    i = 0; count = 0
    while i < N:

        count += 1

        x1 = np.random.uniform()
        x2 = np.random.uniform()
        x3 = 2*np.pi*np.random.uniform()

        r = (x1**(-2./3.)-1.)**(-1./2.)

        if r > r_cut_off:
            continue

        z = (1.-2.*x2)*r
        x = np.sqrt(r*r - z*z)*np.cos(x3)
        y = np.sqrt(r*r - z*z)*np.sin(x3)

        x5 = 0.1
        q  = 0.0
        while x5 > q*q*(1.-q*q)**3.5:

            x5 = 0.1*np.random.uniform()
            q = np.random.uniform()

        ve = np.sqrt(2)*(1. + r*r)**(-1./4.)
        v = q*ve

        x6 = np.random.uniform()
        x7 = 2*np.pi*np.random.uniform()

        vz = (1.-2.*x6)*v
        vx = np.sqrt(v*v - vz*vz)*np.cos(x7)
        vy = np.sqrt(v*v - vz*vz)*np.sin(x7)

        x *= 3.*np.pi/64.*M*M/E
        y *= 3.*np.pi/64.*M*M/E
        z *= 3.*np.pi/64.*M*M/E

        vx *= np.sqrt(E*64./3./np.pi/M)
        vy *= np.sqrt(E*64./3./np.pi/M)
        vz *= np.sqrt(E*64./3./np.pi/M)

        particles["position-x"][i] = x
        particles["position-y"][i] = y
        particles["position-z"][i] = z

        particles["velocity-x"][i] = vx
        particles["velocity-y"][i] = vy
        particles["velocity-z"][i] = vz

        i += 1

    particles["mass"][:] = M/count
    particles["ids"][:] = np.arange(N)

    # center at (50, 50, 50)
    particles["position-x"][:] += 50.
    particles["position-y"][:] += 50.
    particles["position-z"][:] += 50.

if phd._rank == 0:

    num_part = 10000
    particles_root = phd.HydroParticleCreator(num_part, dim=3)

    make_plummer(particles_root, 1000., R=1., r_cut_off=10.)

    # how many particles_root to each process
    nsect, extra = divmod(num_part, phd._size)
    lengths = extra*[nsect+1] + (phd._size-extra)*[nsect]
    send = np.array(lengths)

    # how many particles_root 
    disp = np.zeros(phd._size, dtype=np.int32)
    for i in range(1, phd._size):
        disp[i] = send[i-1] + disp[i-1]

else:

    lengths = disp = send = None
    particles_root = {
            "position-x": None,
            "position-y": None,
            "position-z": None,
            "velocity-x": None,
            "velocity-y": None,
            "velocity-z": None,
            "mass": None,
            "ids": None
            }


# tell each processor how many particles it will hold
send = phd._comm.scatter(send, root=0)

# allocate local particle container
particles = phd.HydroParticleCreator(send, dim=3)

# import particles from root
fields = ["position-x", "position-y", "position-z",
          "velocity-x", "velocity-y", "velocity-z",
          "mass", "ids"]

for field in fields:
    phd._comm.Scatterv([particles_root[field], (lengths, disp)], particles[field])

particles["tag"][:] = phd.ParticleTAGS.Real
particles["type"][:] = phd.ParticleTAGS.Undefined

# computation related to boundaries
domain_manager = phd.DomainManager(
        xmin=[0., 0., 0.], xmax=[100., 100., 100.], initial_radius=0.1, dim=3) 

# load balance
load_balance = phd.LoadBalance(factor=0.1, min_in_leaf=0, order=18)

# setup gravity
gravity_tree = phd.GravityTree(barnes_angle=0.4, smoothing_length=0.03)

# computation
integrator = phd.Nbody(dt=0.005)
integrator.set_particles(particles)
integrator.set_domain_manager(domain_manager)
integrator.set_gravity_tree(gravity_tree)
integrator.set_load_balance(load_balance)

# add finish criteria
simulation_time_manager = phd.SimulationTimeManager()
simulation_time_manager.add_finish(phd.Time(time_max=1.0))

# output first step
output = phd.InitialOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# output last step
output = phd.FinalOutput()
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

output = phd.IterationInterval(iteration_interval=25)
output.set_writer(phd.Hdf5())
simulation_time_manager.add_output(output)

# Create simulator
simulation = phd.Simulation(simulation_name="plummer", colored_logs=True)
simulation.set_integrator(integrator)
simulation.set_simulation_time_manager(simulation_time_manager)
simulation.initialize()
simulation.solve()
