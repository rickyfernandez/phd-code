import numpy as np

def simulation():

    parameters = {
            "CFL" : 0.3,
            "gamma" : 1.4,
            "max_steps" : 1000,
            "max_time" : 0.15,
            "output_name" : "Sod",
            "output_cycle" : 1000,
            "regularization" : True
            }

    gamma = parameters["gamma"]

    L = 1.0
    n = 100

    dx = L/n
    qx = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx

    L = 0.1
    n = 5

    dq = L/n
    q = (np.arange(n+6, dtype=np.float64) - 3)*dq + 0.5*dq

    Nx = qx.size
    Nq = q.size
    N  = Nx*Nq*Nq

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    part = 0
    for i in xrange(Nx):
        for j in xrange(Nq):
            for k in xrange(Nq):
                x[part] = qx[i]
                y[part] = q[j]
                z[part] = q[k]
                part += 1

    # perturb particles to break symmetry 
    indices = ((0.5-dx) <= x) & (x <= (0.5+dx))
    num_points = np.sum(indices)
    x[indices] += 1.0E-3*dx*(2.0*np.random.random(num_points)-1.0)
    y[indices] += 1.0E-3*dq*(2.0*np.random.random(num_points)-1.0)
    z[indices] += 1.0E-3*dq*(2.0*np.random.random(num_points)-1.0)

    num_points = np.sum(~indices)
    y[~indices] += 1.0E-3*dq*(2.0*np.random.random(num_points)-1.0)
    z[~indices] += 1.0E-3*dq*(2.0*np.random.random(num_points)-1.0)
    print "interface has", num_points, "interface points\n"

    # find all particles inside the unit box 
    indices = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= .1)) & ((0. <= z) & (z <= .1)))
    x_in = x[indices]; y_in = y[indices]; z_in = z[indices]

    data = np.zeros((5, x_in.size))
    left_cells = np.where(x_in <= 0.5)[0]
    data[0, left_cells] = 1.0               # density
    data[4, left_cells] = 1.0/(gamma-1.0)   # total energy

    right_cells = np.where(0.5 <= x_in)[0]
    data[0, right_cells] = 0.125            # density
    data[4, right_cells] = 0.1/(gamma-1.0)  # total energy

    # store real particles
    x_particles = np.copy(x_in); y_particles = np.copy(y_in); z_particles = np.copy(z_in)
    particles_index = {"real": np.arange(x_particles.size)}

    # store ghost particles
    x_particles = np.append(x_particles, x[~indices])
    y_particles = np.append(y_particles, y[~indices])
    z_particles = np.append(z_particles, z[~indices])

    # store indices of ghost particles
    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)

    # particle list of real and ghost particles
    particles = np.array([x_particles, y_particles, z_particles])


    return parameters, data, particles, particles_index

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _, data, particles, particles_index = simulation()
    real = particles_index["real"]

    plt.figure(figsize=(8,8))

    plt.subplot(3,1,1)
    plt.scatter(particles[0,real], particles[2,real], facecolors="none", edgecolors="b")
    plt.xlim(0,1)
    plt.ylim(0,.1)
    plt.xlabel("x")
    plt.ylabel("z")

    plt.subplot(3,1,2)
    plt.scatter(particles[0,real], particles[1,real], facecolors="none", edgecolors="b")
    plt.xlim(0,1)
    plt.ylim(0,.1)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(3,1,3)
    plt.scatter(particles[1,real], particles[2,real], facecolors="none", edgecolors="b")
    plt.xlim(0,.1)
    plt.ylim(0,.1)
    plt.xlabel("y")
    plt.ylabel("z")

    plt.tight_layout()
    plt.show()
