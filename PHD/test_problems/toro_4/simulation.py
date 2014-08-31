import numpy as np

def simulation():

    gamma = 1.4

    Lx = 1.       # domain size in x
    Ly = .1       # domain size in y
    nx = 100      # number of points in x
    ny = 10       # number of points in y

    # generate the domain including ghost particles
    dx = Lx/nx
    dy = Ly/ny
    x = (np.arange(nx+6, dtype=np.float64) - 3)*dx + 0.5*dx
    y = (np.arange(ny+6, dtype=np.float64) - 3)*dy + 0.5*dy

    X, Y = np.meshgrid(x,y); Y = np.flipud(Y)

    x = X.flatten(); y = Y.flatten()

    # define domain boundaries 
    left   = 0.0; right = 1.0
    bottom = 0.0; top   = 0.1

    # find all particles in the interior domain
    indices = (((left <= x) & (x <= right)) & ((bottom <= y) & (y <= top)))
    x_in = x[indices]; y_in = y[indices]

    # set initial data - ghost particles do not need to be set
    # since their values are passed for interior particles
    data = np.zeros((4, x_in.size))
    left_cells = np.where(x_in <= 0.5)[0]
    data[0, left_cells] = 1.0                # density
    data[3, left_cells] = 0.01/(gamma-1.0)   # total energy

    right_cells = np.where(0.5 <= x_in)[0]
    data[0, right_cells] = 1.0               # density
    data[3, right_cells] = 100./(gamma-1.0)  # total energy

    # interior particles are real particles
    x_particles = np.copy(x_in); y_particles = np.copy(y_in)
    particles_index = {}

    # store the indices of real particles
    particles_index["real"] = np.arange(x_particles.size)

    # exterior particles are ghost particles
    x_particles = np.append(x_particles, x[~indices])
    y_particles = np.append(y_particles, y[~indices])

    # store the indices of ghost particles
    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)

    # particle list of real and ghost particles
    particles = np.array([x_particles, y_particles])

    return data, particles, particles_index
