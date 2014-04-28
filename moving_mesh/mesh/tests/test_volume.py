import moving_mesh.mesh as mesh
import numpy as np


def test_volume():
    """Test if particle volumes are created correctly. 
    First create a grid of particles in a unit box. So
    the total volume  is 1.0. Then perturb the particles
    in a box of unit lenght 0.5. Create the tessellation
    and compare the sum of all the particle volumes and
    the total volume.
    """

    L = 1.      # box size
    n = 50      # number of points
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

    # create the voronoi diagram
    particles = np.asarray(zip(x,y))
    neighbor_graph, face_graph, voronoi_vertices = mesh.tessellation(particles)

    # find all particles inside the unit box 
    k = np.where(((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 1.)))[0]
    
    # these are the real particles, the rest are ghost particles
    particles_index = {"real": k}
    
    # calculate voronoi volumes of all real particles 
    volume = mesh.volume_center_mass(particles, neighbor_graph, particles_index, face_graph, voronoi_vertices)
    volumes = np.sum(volume[0,:])

    assert np.abs(1.0 - volumes) < 1.0E-10
