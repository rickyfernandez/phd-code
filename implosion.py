import numpy as np

def implosion():

    # boundaries
    boundary_dic = {"left":0.0, "right":1.0, "bottom":0.0, "top":1.0}

    L = 1.       # domain size
    n = 50      # number of points
    gamma = 1.4

    dx = L/n
    x = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
    X, Y = np.meshgrid(x,x); Y = np.flipud(Y)

    x = X.flatten(); y = Y.flatten()

    left   = boundary_dic["left"];   right = boundary_dic["right"]
    bottom = boundary_dic["bottom"]; top   = boundary_dic["top"]

    indices = (((left <= x) & (x <= right)) & ((bottom <= y) & (y <= top)))
    x_in = x[indices]; y_in = y[indices]

    indices_in = (x_in+y_in) > 0.5
    data = np.zeros((4, x_in.size))
    data[0, indices_in] = 1.0
    data[3, indices_in] = 1.0/(gamma-1.0)

    data[0, ~indices_in] = 0.125
    data[3, ~indices_in] = 0.14/(gamma-1.0)

    x_particles = np.copy(x_in); y_particles = np.copy(y_in)
    particles_index = {}
    particles_index["real"] = np.arange(x_particles.size)

    x_particles = np.append(x_particles, x[~indices])
    y_particles = np.append(y_particles, y[~indices])
    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)

    particles = np.array([x_particles, y_particles])

    return data, particles, particles_index

#-------------------------------------------------------------------------------
import PHD.simulation as simulation
import PHD.boundary as boundary
import PHD.riemann as riemann

# parameters for the simulation
CFL = 0.5
gamma = 1.4
max_steps = 200
max_time = 0.5
output_name = "Implosion_"

# create boundary and riemann objects
boundary_condition = boundary.reflect(0.,1.,0.,1.)
riemann_solver = riemann.pvrs()

# create initial state of the system
data, particles, particles_index = implosion()

# setup the moving mesh simulation
simulation = simulation.moving_mesh()

# set runtime parameters for the simulation
simulation.set_parameter("CFL", CFL)
simulation.set_parameter("gamma", gamma)
simulation.set_parameter("max_steps", max_steps)
simulation.set_parameter("max_time", max_time)
simulation.set_parameter("output_name", output_name)

# set the boundary, riemann solver, and initial state of the simulation 
simulation.set_boundary_condition(boundary_condition)
simulation.set_riemann_solver(riemann_solver)
simulation.set_initial_state(particles, data, particles_index)

# run the simulation
simulation.solve()
