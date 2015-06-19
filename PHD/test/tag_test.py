L = 1.      # box size
n = 50      # number of points
dx = L/n

# add ghost 3 ghost particles to the sides for the tesselation
# wont suffer from edge boundaries
x = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx

# generate the grid of particle positions
X, Y = np.meshgrid(x,x); Y = np.flipud(Y)
x = X.flatten(); y = Y.flatten()

# find all particles inside the unit box 
indices = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 1.)))
x_in = x[indices]; y_in = y[indices]

# find particles in the interior box
k = ((0.25 < x_in) & (x_in < 0.5)) & ((0.25 < y_in) & (y_in < 0.5))

# randomly perturb their positions
num_points = k.sum()
x_in[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)
y_in[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)

# store real particles
x_particles = np.copy(x_in); y_particles = np.copy(y_in)
particles_index = {"real": np.arange(x_particles.size)}

# store ghost particles
x_particles = np.append(x_particles, x[~indices])
y_particles = np.append(y_particles, y[~indices])

# store indices of ghost particles
particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)

# particle list of real and ghost particles
particles = np.array([x_particles, y_particles])

pc = ParticleContainer(x_particles.size)
pc['position-x'][:] = x_particles[:]
pc['position-y'][:] = y_particles[:]

# generate voronoi mesh 
mesh = VoronoiMesh2D()
graphs = mesh.tessellate(particles)

