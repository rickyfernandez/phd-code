import numpy as np
import PHD.boundary as boundary
import PHD.mesh as mesh

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection

def test_reflect2d_boundary():
    """
    Test if reflection boundary picks the right real particles for creating ghost
    particles
    """

    L = 1.      # box size
    n = 10      # number of points
    dx = L/n

    # add ghost 3 ghost particles to the sides for the initial tesselation
    x = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx

    # generate the grid of particle positions
    X, Y = np.meshgrid(x,x); Y = np.flipud(Y)
    x = X.flatten(); y = Y.flatten()

    # find all particles inside the unit box, these are real
    indices = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 1.)))
    x_in = x[indices]; y_in = y[indices]

    # perturb real particle positions
    num_points = x_in.size
    x_in += 0.2*dx*(2.0*np.random.random(num_points)-1.0)
    y_in += 0.2*dx*(2.0*np.random.random(num_points)-1.0)

    # store real particles
    x_particles = np.copy(x_in); y_particles = np.copy(y_in)

    # indices of real particles
    particles_index = {"real": np.arange(x_particles.size)}

    # store ghost particles
    x_particles = np.append(x_particles, x[~indices])
    y_particles = np.append(y_particles, y[~indices])

    # indices of ghost particles
    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)

    # particle array 
    particles = np.array([x_particles, y_particles])

    # generate voronoi mesh to generate graphs 
    m = mesh.VoronoiMesh2D()
    graphs = m.tessellate(particles)

    # create ghost particels 
    reflect = boundary.Reflect2D(0.,1.,0.,1.)
    particles = reflect.update_boundaries(particles, particles_index, graphs["neighbors"], graphs["number of neighbors"])

    # update the graphs with the new ghost particles
    graphs = m.tessellate(particles)

    # brute force find the indices of real particles that will be used to make ghost particles

    x = particles[0,:]; y = particles[1,:]

    # store the indices
    s = set()

    # grab left boundary, two layers
    k = np.where(((0.0 < x) & (x < 2.0*dx)) & ((0.0 < y) & (y < 1.0)))[0]
    s.update(k)

    # grab right boundary, two layers
    k = np.where((((1.0-2.0*dx) < x) & (x < 1.0)) & ((0.0 < y) & (y < 1.0)))[0]
    s.update(k)

    # grab bottom boundary, two layers
    k = np.where(((0.0 < y) & (y < 2.0*dx)) & ((0.0 < x) & (x < 1.0)))[0]
    s.update(k)

    # grab top boundary, two layers
    k = np.where((((1.0-2.0*dx) < y) & (y < 1.0)) & ((0.0 < x) & (x < 1.0)))[0]
    s.update(k)

    # plot the comparison to visually inspect
    l = []
    ii = 0; jj = 0
    for ip in particles_index["real"]:

        # plot each cell
        jj += graphs["number of neighbors"][ip]*2
        verts_indices = np.unique(graphs["faces"][ii:jj])
        verts = graphs["voronoi vertices"][verts_indices]

        # coordinates of neighbors relative to particle p
        xc = verts[:,0] - particles[0,ip]
        yc = verts[:,1] - particles[1,ip]

        # sort in counter clock wise order
        sorted_vertices = np.argsort(np.angle(xc+1j*yc))
        verts = verts[sorted_vertices]

        l.append(Polygon(verts, True))

        ii = jj

    fig, ax = plt.subplots(figsize=(8, 8))
    p = PatchCollection(l, alpha=0.4)
    ax.add_collection(p)

    # plot ghost particles
    ghost_indices = particles_index["ghost"]
    plt.plot(particles[0,ghost_indices], particles[1,ghost_indices], 'ro')

    # go throught ghost particles and plot their index and the index of the corresponding real
    # particle
    ghost_map = particles_index["ghost_map"]
    for p in ghost_indices:

        ghost_point = particles[:,p]
        plt.text(ghost_point[0], ghost_point[1], '%d' % ghost_map[p], color='b', ha='center')

        real_point = particles[:,ghost_map[p]]
        plt.text(real_point[0], real_point[1]+0, '%d' % ghost_map[p], color='k', ha='center')

    plt.savefig("boundary")

    # make shure indices picked from boundary class match brute force calculation 
    ghost_map = particles_index["ghost_map"]
    map_indices = [ghost_map[i] for i in particles_index["ghost"]]

    assert set(map_indices).issubset(s)


def test_reflect3d_boundary():
    """
    Test if reflection boundary picks the right real particles for creating ghost
    particles
    """

    L = 1.0
    n = 5

    dx = L/n
    q = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx

    N = q.size
    x = np.zeros(N**3)
    y = np.zeros(N**3)
    z = np.zeros(N**3)

    part = 0
    for i in xrange(N):
        for j in xrange(N):
            for k in xrange(N):
                x[part] = q[i]
                y[part] = q[j]
                z[part] = q[k]
                part += 1


    # find all particles inside the unit box 
    indices = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 1.)) & ((0. <= z) & (z <= 1.)))
    x_in = x[indices]; y_in = y[indices]; z_in = z[indices]

    # find particles in the interior box
    k = (((0.25 < x_in) & (x_in < 0.5)) & ((0.25 < y_in) & (y_in < 0.5)) & ((0.25 < z_in) & (z_in < 0.5)))

    # randomly perturb their positions
    num_points = k.sum()
    x_in[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)
    y_in[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)
    z_in[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)

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

    # generate voronoi mesh to generate graphs 
    m = mesh.VoronoiMesh3D()
    graphs = m.tessellate(particles)

    # create ghost particels 
    reflect = boundary.Reflect3D(0.,1.,0.,1.,0.,1.)
    particles = reflect.update_boundaries(particles, particles_index, graphs["neighbors"], graphs["number of neighbors"])

    # update the graphs with the new ghost particles
    graphs = m.tessellate(particles)

    # brute force find the indices of real particles that will be used to make ghost particles

    x = particles[0,:]; y = particles[1,:]; z = particles[2,:]

    # store the indices
    s = set()

    # grab left boundary, two layers
    k = np.where(((0.0 < x) & (x < 2.0*dx)) & ((0.0 < y) & (y < 1.0)) & ((0.0 < z) & (z < 1.0)))[0]
    s.update(k)

    # grab right boundary, two layers
    k = np.where((((1.0-2.0*dx) < x) & (x < 1.0)) & ((0.0 < y) & (y < 1.0)) & ((z < 1.0) & (z < 1.0)))[0]
    s.update(k)

    # grab bottom boundary, two layers
    k = np.where(((0.0 < y) & (y < 2.0*dx)) & ((0.0 < x) & (x < 1.0)) & ((0.0 < z) & (z < 1.0)))[0]
    s.update(k)

    # grab top boundary, two layers
    k = np.where((((1.0-2.0*dx) < y) & (y < 1.0)) & ((0.0 < x) & (x < 1.0)) & ((0.0 < z) & (z < 1.0)))[0]
    s.update(k)

    # grab bottom boundary, two layers
    k = np.where(((0.0 < z) & (z < 2.0*dx)) & ((0.0 < x) & (x < 1.0)) & ((0.0 < y) & (y < 1.0)))[0]
    s.update(k)

    # grab top boundary, two layers
    k = np.where((((1.0-2.0*dx) < z) & (z < 1.0)) & ((0.0 < x) & (x < 1.0)) & ((0.0 < y) & (y < 1.0)))[0]
    s.update(k)

    # make shure indices picked from boundary class match brute force calculation 
    ghost_map = particles_index["ghost_map"]
    map_indices = [ghost_map[i] for i in particles_index["ghost"]]

    assert set(map_indices).issubset(s)

if __name__ == "__main__":
    test_boundary()
