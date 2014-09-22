import numpy as np

def simulation():


    parameters = {
            "CFL" : 0.3,
            "gamma" : 1.4,
            "max_steps" : 1000,
            "max_time" : 0.10,
            "output_name" : "sedov",
            "output_cycle" : 1000,
            "regularization" : True
            }

    gamma = parameters["gamma"]

    Lx = 1.0      # domain size in x
    Ly = 1.0      # domain size in y
    nx = 51       # number of points in x
    ny = 51       # number of points in y

    # generate the domain including ghost particles
    dx = Lx/nx
    dy = Ly/ny
    x = (np.arange(nx+6, dtype=np.float64) - 3)*dx + 0.5*dx
    y = (np.arange(ny+6, dtype=np.float64) - 3)*dy + 0.5*dy

    X, Y = np.meshgrid(x,y); Y = np.flipud(Y)

    x = X.flatten(); y = Y.flatten()

    # define domain boundaries 
    left   = 0.0; right = 1.0
    bottom = 0.0; top   = 1.0

    # find all particles in the interior domain
    indices = (((left < x) & (x < right)) & ((bottom < y) & (y < top)))
    x_in = x[indices]; y_in = y[indices]

    data = np.zeros((4, x_in.size))

    # set ambient values
    data[0,:] = 1.0                    # density
    data[3,:] = 1.0E-5                 # energy density

    r = 0.01
    cells = ((x_in-.5)**2 + (y_in-.5)**2) <= r**2
    #data[3, cells] = 1.0/(np.pi*r**2)
    data[3, cells] = 1.0/(dx*dy)
    print "number of cells:", np.sum(cells)

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

    return parameters, data, particles, particles_index
