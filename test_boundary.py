import PHD.mesh.cell_volume_center as cv
import PHD.boundary.boundary_base as boundary_base
import PHD.mesh as mesh
import numpy as np

def boundary():

    L = 1.      # box size
    n = 50    # number of points
    dx = L/n

    # add ghost 3 ghost particles to the sides for the tesselation
    # wont suffer from edge boundaries
    x = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx

    # generate the grid of particle positions
    X, Y = np.meshgrid(x,x); Y = np.flipud(Y)
    x = X.flatten(); y = Y.flatten()

    # find particles in the interior box
    k = ((0.25 < x) & (x < 0.5)) & ((0.25 < y) & (y < 0.5))

    # randomly perturb their positions
    num_points = k.sum()
    x[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)
    y[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)

    # find all particles inside the unit box 
    k = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 1.)))
    
    # stack particles
    x_particles = np.copy(x[k]); y_particles = np.copy(y[k])
    particles_index = {}
    particles_index["real"] = np.arange(x_particles.size, dtype=np.int32)

    x_particles = np.append(x_particles, x[~k])
    y_particles = np.append(y_particles, y[~k])
    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size, dtype=np.int32)

    # create the voronoi diagram
    particles = np.asarray(zip(x_particles,y_particles))
    vor_mesh = mesh.voronoi_mesh()
    neighbor_graph, face_graph, voronoi_vertices, ng, ngs, fg, fgs = vor_mesh.tessellate(particles)

    bb = boundary_base.boundary_base()

    ghost_indices = particles_index["ghost"]
    i = np.where(1.0 > particles[ghost_indices,0])
    x1 = bb.find_boundary_particles(neighbor_graph, ghost_indices[i], ghost_indices)
    x2 = bb.find_boundary_particles2(ng, ngs, ghost_indices[i], ghost_indices)

    return x1 == x2

if __name__ == '__main__':
    x = boundary()
    print x.sum()/x.size
