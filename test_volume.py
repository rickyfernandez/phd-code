import PHD.mesh.cell_volume_center as cv
import PHD.mesh as mesh
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
    vor_mesh = mesh.voronoi_mesh()
    neighbor_graph, face_graph, voronoi_vertices, _, _, _, _ = vor_mesh.tessellate(particles)

    # find all particles inside the unit box 
    k = np.where(((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 1.)))[0]
    
    # these are the real particles, the rest are ghost particles
    particles_index = {"real": k}
    
    # calculate voronoi volumes of all real particles 
    volume = vor_mesh.volume_center_mass(particles, neighbor_graph, particles_index,
            face_graph, voronoi_vertices)
    volumes = np.sum(volume[0,:])
    print volumes
    #assert np.abs(1.0 - volumes) < 1.0E-10

def test_volume2():
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
    
    # calculate voronoi volumes of all real particles 
    volume = vor_mesh.volume_center_mass(particles, neighbor_graph, particles_index,
            face_graph, voronoi_vertices)

    print "hello world"
    volume2 = np.zeros(len(k), dtype=np.float64)
    center_of_mass = np.zeros((2,len(k)), dtype=np.float64)

    import PHD.mesh.cell_volume_center as cv
    import PHD.mesh.test as t
    #print t.get_val(len(k))
    #print fg.dtype
    #t.loop_int_array(fg, fg.size)
    cv.cell_volume_center(particles, ng, ngs, fg, voronoi_vertices, volume2,
            center_of_mass, particles_index["real"].size)
    #cv.cell_volume_center(particles, ng, ngs, fg, voronoi_vertices, volume2, center_of_mass, 2)

    #volumes = np.sum(volume)
#
    print "volume in mesh:", volume[0,:].sum()
    print "volume in cython:", volume2.sum()
    #assert np.abs(1.0 - volumes) < 1.0E-10

def setup_test():
    """Test if particle volumes are created correctly. 
    First create a grid of particles in a unit box. So
    the total volume  is 1.0. Then perturb the particles
    in a box of unit lenght 0.5. Create the tessellation
    and compare the sum of all the particle volumes and
    the total volume.
    """

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
    return vor_mesh, particles, particles_index, neighbor_graph, face_graph, voronoi_vertices, ng, ngs, fg, fgs
    
def numpy_volume(vor_mesh, particles, neighbor_graph, particles_index, face_graph, voronoi_vertices):
#    # calculate voronoi volumes of all real particles 
    volume = vor_mesh.volume_center_mass(particles, neighbor_graph, particles_index,
            face_graph, voronoi_vertices)

    print "volume in mesh:", volume[0,:].sum()

def numpy_cython_volume():

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
   
    # numpy volume and center mass calculation
    volume = vor_mesh.volume_center_mass(particles, neighbor_graph, particles_index,
            face_graph, voronoi_vertices)


    # cython volume and center mass calculation
    k = particles_index["real"].size
    volume2 = np.zeros(k, dtype=np.float64)
    center_of_mass = np.zeros((2,k), dtype=np.float64)
    cv.cell_volume_center(particles, ng, ngs, fg, voronoi_vertices, volume2,
            center_of_mass, k)

    print "volume in mesh:", volume[0,:].sum()
    print "com in mesh:   ", volume[1:3, 0:5]
    print "volume in cython:", volume2.sum()
    print "com in cython:   ", center_of_mass[0:2, 0:5]
    print "com diff: %0.15e" % np.max(center_of_mass - volume[1:3,:])

#def cython_volume():
#    """Test if particle volumes are created correctly. 
#    First create a grid of particles in a unit box. So
#    the total volume  is 1.0. Then perturb the particles
#    in a box of unit lenght 0.5. Create the tessellation
#    and compare the sum of all the particle volumes and
#    the total volume.
#    """
#
#    L = 1.      # box size
#    n = 50    # number of points
#    dx = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    x = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
#
#    # generate the grid of particle positions
#    X, Y = np.meshgrid(x,x); Y = np.flipud(Y)
#    x = X.flatten(); y = Y.flatten()
#
#    # find particles in the interior box
#    k = ((0.25 < x) & (x < 0.5)) & ((0.25 < y) & (y < 0.5))
#
#    # randomly perturb their positions
#    num_points = k.sum()
#    x[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)
#    y[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)
#
#    # find all particles inside the unit box 
#    k = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 1.)))
#    
#    # stack particles
#    x_particles = np.copy(x[k]); y_particles = np.copy(y[k])
#    particles_index = {}
#    particles_index["real"] = np.arange(x_particles.size, dtype=np.int32)
#
#    x_particles = np.append(x_particles, x[~k])
#    y_particles = np.append(y_particles, y[~k])
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size, dtype=np.int32)
#
#    # create the voronoi diagram
#    particles = np.asarray(zip(x_particles,y_particles))
#    vor_mesh = mesh.voronoi_mesh()
#    neighbor_graph, face_graph, voronoi_vertices, ng, ngs, fg, fgs = vor_mesh.tessellate(particles)
#    neighbor_graph, face_graph, voronoi_vertices, ng, ngs, fg, fgs = vor_mesh.tessellate(particles)
#    
#
def cython_volume(particles, ng, ngs, fg, voronoi_vertices, particles_index):

    k = particles_index["real"].size
    volume = np.zeros(k, dtype=np.float64)
    center_of_mass = np.zeros((2,k), dtype=np.float64)
    cv.cell_volume_center(particles, ng, ngs, fg, voronoi_vertices, volume,
            center_of_mass, k)

    print "volume in cython:", volume.sum()

if __name__ == '__main__':
    numpy_cython_volume()
