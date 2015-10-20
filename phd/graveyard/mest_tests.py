import unittest
import numpy as np

from phd.mesh.voronoi_mesh import VoronoiMesh2D
from utils.particle_tags import ParticleTAGS
from phd.containers.containers import ParticleContainer


class TestVoronoiMesh2dBox(unittest.TestCase):

    def setUp(self):

        L = 1.      # box size
        n = 50      # number of points
        self.dx = L/n

        # add ghost 3 ghost particles to the sides for the tesselation
        # wont suffer from edge boundaries
        x = (np.arange(n+6, dtype=np.float64) - 3)*self.dx + 0.5*self.dx

        # generate the grid of particle positions
        X, Y = np.meshgrid(x,x); Y = np.flipud(Y)
        x = X.flatten(); y = Y.flatten()

        # find all particles inside the unit box 
        indices = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 1.)))
        x_in = x[indices]; y_in = y[indices]
        self.num_real = x_in.size

        self.particles = ParticleContainer(x_in.size)

        # store ghost particles
        xp = self.particles["position-x"]; yp = self.particles["position-y"]
        xp[:] = x_in; yp[:] = y_in

        # store ghost particles
        x_out = x[~indices]; y_out = y[~indices]
        self.num_ghost = x_out.size
        self.particles.extend(x_out.size)

        tag = self.particles["tag"]
        xp = self.particles["position-x"]; yp = self.particles["position-y"]
        xp[self.num_real:] = x_out; yp[self.num_real:] = y_out
        tag[self.num_real:] = ParticleTAGS.Ghost

        # put real particles in front of the arrays
#        type = self.particles["tag"]
#        type[:] = ParticleTAGS.Ghost
#        self.particles.align_particles()
#        vol = self.particles["volume"]
#        vol[:] = 0.0


    def test_volume_2D(self):
        """Test particle volumes in square created correctly.
        Create grid of particles in a unit box, total volume
        is 1.0. Create tessellation and sum all particle
        volumes.
        """
        pos = np.array([
            self.particles["position-x"],
            self.particles["position-y"]
            ])

        # generate voronoi mesh 
        mesh = VoronoiMesh2D()
        mesh.tessellate(pos)

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info(self.particles)
        real_indices = self.particles["tag"] == ParticleTAGS.Real
        tot_vol = np.sum(self.particles["volume"][real_indices])

        self.assertAlmostEqual(tot_vol, 1.0)

    def test_volume_perturb_2D(self):
        """Test particle volumes in a perturb sub-square are created correctly.
        Create grid of particles in a unit box, and perturb the positions
        in a sub-square. Total volume is 1.0. Create tessellation and sum all
        particle volumes.
        """

        # find particles in the interior box
        x_in = self.particles["position-x"]; y_in = self.particles["position-y"]
        k = ((0.25 < x_in) & (x_in < 0.5)) & ((0.25 < y_in) & (y_in < 0.5))

        # randomly perturb their positions
        num_points = k.sum()
        x_in[k] += 0.2*self.dx*(2.0*np.random.random(num_points)-1.0)
        y_in[k] += 0.2*self.dx*(2.0*np.random.random(num_points)-1.0)

        pos = np.array([
            self.particles["position-x"],
            self.particles["position-y"]
            ])

        # generate voronoi mesh 
        mesh = VoronoiMesh2D()
        mesh.tessellate(pos)

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info(self.particles)
        real_indices = self.particles["tag"] == ParticleTAGS.Real
        tot_vol = np.sum(self.particles["volume"][real_indices])

        self.assertAlmostEqual(tot_vol, 1.0)

    def test_center_of_mass_2D(self):
        """Test if particle center of mass positions are created correctly.
        Particles or in a uniform unit box. So com is just the particle
        positions.
        """
        pos = np.array([
            self.particles["position-x"],
            self.particles["position-y"]
            ])

        # generate voronoi mesh 
        mesh = VoronoiMesh2D()
        mesh.tessellate(pos)

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info(self.particles)
        real_indices = self.particles["tag"] == ParticleTAGS.Real

        xcom = self.particles["com-x"][real_indices];    ycom = self.particles["com-y"][real_indices]
        xp = self.particles["position-x"][real_indices]; yp = self.particles["position-y"][real_indices]

        for i in range(xp.size):
            self.assertAlmostEqual(xcom[i], xp[i])
            self.assertAlmostEqual(ycom[i], yp[i])


class TestVoronoiMesh2dRectangle(unittest.TestCase):

    def setUp(self):

        xvals = -2*np.random.random(2) + 1
        self.xmin = np.min(xvals)
        self.xmax = np.max(xvals)

        L = (self.xmax-self.xmin)  # size in x
        n = 50                     # number of points along dimension
        self.dx = L/n

        # add ghost 3 ghost particles to the sides for the tesselation
        # wont suffer from edge boundaries
        x = self.xmin + (np.arange(n+6, dtype=np.float64) - 3)*self.dx + 0.5*self.dx

        yvals = -2*np.random.random(2) + 1
        self.ymin = np.min(yvals)
        self.ymax = np.max(yvals)

        L = (self.ymax-self.ymin)  # size in x
        n = 50                     # number of points along dimension
        self.dy = L/n

        # add ghost 3 ghost particles to the sides for the tesselation
        # wont suffer from edge boundaries
        y = self.ymin + (np.arange(n+6, dtype=np.float64) - 3)*self.dy + 0.5*self.dy

        # generate the grid of particle positions
        X, Y = np.meshgrid(x,y); Y = np.flipud(Y)
        x = X.flatten(); y = Y.flatten()

        # find all particles inside the unit box 
        indices = (((self.xmin <= x) & (x <= self.xmax)) & ((self.ymin <= y) & (y <= self.ymax)))
        x_in = x[indices]; y_in = y[indices]
        self.num_real = x_in.size

        self.particles = ParticleContainer(x_in.size)

        # store ghost particles
        xp = self.particles["position-x"]; yp = self.particles["position-y"]
        xp[:] = x_in; yp[:] = y_in

        # store ghost particles
        x_out = x[~indices]; y_out = y[~indices]
        self.num_ghost = x_out.size
        self.particles.extend(x_out.size)

        tag = self.particles["tag"]
        xp = self.particles["position-x"]; yp = self.particles["position-y"]
        xp[self.num_real:] = x_out; yp[self.num_real:] = y_out
        tag[self.num_real:] = ParticleTAGS.Ghost

        # put real particles in front of the arrays
#        type = self.particles["tag"]
#        type[:] = ParticleTAGS.Ghost
#        self.particles.align_particles()
#        vol = self.particles["volume"]
#        vol[:] = 0.0


    def test_volume_2D(self):
        """Test particle volumes in random rectangle are created correctly.
        Create grid of particles in a random size rectangle. Create
        tessellation and sum all particle volumes.
        """
        pos = np.array([
            self.particles["position-x"],
            self.particles["position-y"]
            ])

        # generate voronoi mesh 
        mesh = VoronoiMesh2D()
        mesh.tessellate(pos)

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info(self.particles)
        real_indices = self.particles["tag"] == ParticleTAGS.Real
        tot_vol = np.sum(self.particles["volume"][real_indices])

        self.assertAlmostEqual(tot_vol, ((self.xmax-self.xmin)*(self.ymax-self.ymin)))

    def test_volume_perturb_2D(self):
        """Test if random particle volumes are created correctly.
        First create a grid of particles in a unit box. So
        the total volume  is 1.0. Then perturb the particles
        in a box of unit lenght 0.5. Create the tessellation
        and compare the sum of all the particle volumes and
        the total volume.
        """

        Lx = self.xmax - self.xmin
        Ly = self.ymax - self.ymin

        xlo = self.xmin + 0.25*Lx; xhi = self.xmin + 0.75*Lx
        ylo = self.ymin + 0.25*Ly; yhi = self.ymin + 0.75*Ly

        # find particles in the interior box
        x_in = self.particles["position-x"]; y_in = self.particles["position-y"]
        k = ((xlo < x_in) & (x_in < xhi)) & ((ylo < y_in) & (y_in < yhi))

        # randomly perturb their positions
        num_points = k.sum()
        x_in[k] += 0.2*self.dx*(2.0*np.random.random(num_points)-1.0)
        y_in[k] += 0.2*self.dy*(2.0*np.random.random(num_points)-1.0)

        pos = np.array([
            self.particles["position-x"],
            self.particles["position-y"]
            ])

        # generate voronoi mesh 
        mesh = VoronoiMesh2D()
        mesh.tessellate(pos)

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info(self.particles)
        real_indices = self.particles["tag"] == ParticleTAGS.Real
        tot_vol = np.sum(self.particles["volume"][real_indices])

        self.assertAlmostEqual(tot_vol, Lx*Ly)

    def test_center_of_mass_2D(self):
        """Test if particle center of mass positions are created correctly.
        Particles are placed equally spaced in each direction. So com is
        just the particle positions.
        """
        pos = np.array([
            self.particles["position-x"],
            self.particles["position-y"]
            ])

        # generate voronoi mesh 
        mesh = VoronoiMesh2D()
        mesh.tessellate(pos)

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info(self.particles)
        real_indices = self.particles["tag"] == ParticleTAGS.Real

        xcom = self.particles["com-x"][real_indices];    ycom = self.particles["com-y"][real_indices]
        xp = self.particles["position-x"][real_indices]; yp = self.particles["position-y"][real_indices]

        for i in range(xp.size):
            self.assertAlmostEqual(xcom[i], xp[i])
            self.assertAlmostEqual(ycom[i], yp[i])





#def test_volume_random_rectangel_2D():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a rectangular
#    box [0,10]x[0,1], so the total volume is 10.0
#    """
#
#    xvals = -2*np.random.random(2) + 1
#    xmin = np.min(xvals)
#    xmax = np.max(xvals)
#
#    L = (xmax-xmin)  # size in x
#    n = 50           # number of points along dimension
#    dx = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    x = xmin + (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
#
#    yvals = -2*np.random.random(2) + 1
#    ymin = np.min(yvals)
#    ymax = np.max(yvals)
#
#    L = (ymax-ymin)  # size in x
#    n = 50      # number of points along dimension
#    dy = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    y = ymin + (np.arange(n+6, dtype=np.float64) - 3)*dy + 0.5*dy
#
#    # generate the grid of particle positions
#    X, Y = np.meshgrid(x,y); Y = np.flipud(Y)
#    x = X.flatten(); y = Y.flatten()
#
#    # find all particles inside the unit box 
#    indices = (((xmin <= x) & (x <= xmax)) & ((ymin <= y) & (y <= ymax)))
#    x_in = x[indices]; y_in = y[indices]
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh2D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    tot_vol = cells_info["volume"].sum()
#    print "cell volume:", cells_info["volume"][0]
#    print "tot volume:", tot_vol
#    assert np.abs((xmax-xmin)*(ymax-ymin) - tot_vol) < 1.0E-10
#
#def test_volume_x_2D():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a rectangular
#    box [0,10]x[0,1], so the total volume is 10.0
#    """
#
#    L = 10.     # size in x
#    n = 50      # number of points along dimension
#    dx = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    x = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
#
#    L = 1.      # size in y
#    n = 50      # number of points along dimension
#    dy = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    y = (np.arange(n+6, dtype=np.float64) - 3)*dy + 0.5*dy
#
#    # generate the grid of particle positions
#    X, Y = np.meshgrid(x,y); Y = np.flipud(Y)
#    x = X.flatten(); y = Y.flatten()
#
#    # find all particles inside the unit box 
#    indices = (((0. <= x) & (x <= 10.)) & ((0. <= y) & (y <= 1.)))
#    x_in = x[indices]; y_in = y[indices]
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh2D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    tot_vol = cells_info["volume"].sum()
#    print "cell volume:", cells_info["volume"][0]
#    print "tot volume:", tot_vol
#    assert np.abs(10.0 - tot_vol) < 1.0E-10
#
#def test_volume_y_2D():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a rectangular
#    box [0,1]x[0,10], so the total volume is 10.0
#    """
#
#    L = 1.      # size in x
#    n = 50      # number of points along dimension
#    dx = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    x = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
#
#    L = 10.     # size in y
#    n = 50      # number of points along dimension
#    dy = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    y = (np.arange(n+6, dtype=np.float64) - 3)*dy + 0.5*dy
#
#    # generate the grid of particle positions
#    X, Y = np.meshgrid(x,y); Y = np.flipud(Y)
#    x = X.flatten(); y = Y.flatten()
#
#    # find all particles inside the unit box 
#    indices = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 10.)))
#    x_in = x[indices]; y_in = y[indices]
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh2D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    tot_vol = cells_info["volume"].sum()
#    print "cell volume:", cells_info["volume"][0]
#    print "tot volume:", tot_vol
#    assert np.abs(10.0 - tot_vol) < 1.0E-10
#
#def test_center_of_mass_2D():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a unit box. So
#    the total volume  is 1.0. Then perturb the particles
#    in a box of unit lenght 0.5. Create the tessellation
#    and compare the sum of all the particle volumes and
#    the total volume.
#    """
#
#    L = 1.      # box size
#    n = 50      # number of points
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
#    # find all particles inside the unit box 
#    indices = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 1.)))
#    x_in = x[indices]; y_in = y[indices]
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh2D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    com = cells_info["center of mass"]
#
#    # the center of mass of each particle should be their position
#    real = particles_index["real"]
#    diff = np.absolute(particles[:,real] - com[:,:]) < 1.0E-10
#
#    assert np.all(diff)
#
#def test_center_of_mass_x_2d():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a rectangular
#    box [0,10]x[0,1], so the total volume is 10.0
#    """
#
#    L = 10.     # size in x
#    n = 50      # number of points along dimension
#    dx = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    x = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
#
#    L = 1.      # size in y
#    n = 50      # number of points along dimension
#    dy = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    y = (np.arange(n+6, dtype=np.float64) - 3)*dy + 0.5*dy
#
#    # generate the grid of particle positions
#    X, Y = np.meshgrid(x,y); Y = np.flipud(Y)
#    x = X.flatten(); y = Y.flatten()
#
#    # find all particles inside the unit box 
#    indices = (((0. <= x) & (x <= 10.)) & ((0. <= y) & (y <= 1.)))
#    x_in = x[indices]; y_in = y[indices]
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh2D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    com = cells_info["center of mass"]
#
#    # the center of mass of each particle should be their position
#    real = particles_index["real"]
#    diff = np.absolute(particles[:,real] - com[:,:]) < 1.0E-10
#
#    assert np.all(diff)
#
#def test_center_of_mass_y_2D():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a rectangular
#    box [0,1]x[0,10], so the total volume is 10.0
#    """
#
#    L = 1.      # size in x
#    n = 50      # number of points along dimension
#    dx = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    x = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
#
#    L = 10.     # size in y
#    n = 50      # number of points along dimension
#    dy = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    y = (np.arange(n+6, dtype=np.float64) - 3)*dy + 0.5*dy
#
#    # generate the grid of particle positions
#    X, Y = np.meshgrid(x,y); Y = np.flipud(Y)
#    x = X.flatten(); y = Y.flatten()
#
#    # find all particles inside the unit box 
#    indices = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 10.)))
#    x_in = x[indices]; y_in = y[indices]
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh2D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    com = cells_info["center of mass"]
#
#    # the center of mass of each particle should be their position
#    real = particles_index["real"]
#    diff = np.absolute(particles[:,real] - com[:,:]) < 1.0E-10
#
#    assert np.all(diff)
#
#def test_center_of_mass_random_rectangel_2D():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a rectangular
#    box [0,10]x[0,1], so the total volume is 10.0
#    """
#
#    xvals = -2*np.random.random(2) + 1
#    xmin = np.min(xvals)
#    xmax = np.max(xvals)
#
#    L = (xmax-xmin)  # size in x
#    n = 50           # number of points along dimension
#    dx = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    x = xmin + (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
#
#    yvals = -2*np.random.random(2) + 1
#    ymin = np.min(yvals)
#    ymax = np.max(yvals)
#
#    L = (ymax-ymin)  # size in x
#    n = 50      # number of points along dimension
#    dy = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    y = ymin + (np.arange(n+6, dtype=np.float64) - 3)*dy + 0.5*dy
#
#    # generate the grid of particle positions
#    X, Y = np.meshgrid(x,y); Y = np.flipud(Y)
#    x = X.flatten(); y = Y.flatten()
#
#    # find all particles inside the unit box 
#    indices = (((xmin <= x) & (x <= xmax)) & ((ymin <= y) & (y <= ymax)))
#    x_in = x[indices]; y_in = y[indices]
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh2D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    com = cells_info["center of mass"]
#
#    # the center of mass of each particle should be their position
#    real = particles_index["real"]
#    diff = np.absolute(particles[:,real] - com[:,:]) < 1.0E-10
#
#    assert np.all(diff)
#
#def test_volume_perturb_3D():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a unit box. So
#    the total volume  is 1.0. Then perturb the particles
#    in a box of unit lenght 0.5. Create the tessellation
#    and compare the sum of all the particle volumes and
#    the total volume.
#    """
#    L = 1.0
#    n = 5
#
#    dx = L/n
#    q = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
#
#    N = q.size
#    x = np.zeros(N**3)
#    y = np.zeros(N**3)
#    z = np.zeros(N**3)
#
#    part = 0
#    for i in xrange(N):
#        for j in xrange(N):
#            for k in xrange(N):
#                x[part] = q[i]
#                y[part] = q[j]
#                z[part] = q[k]
#                part += 1
#
#
#    # find all particles inside the unit box 
#    indices = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 1.)) & ((0. <= z) & (z <= 1.)))
#    x_in = x[indices]; y_in = y[indices]; z_in = z[indices]
#
#    # find particles in the interior box
#    k = (((0.25 < x_in) & (x_in < 0.5)) & ((0.25 < y_in) & (y_in < 0.5)) & ((0.25 < z_in) & (z_in < 0.5)))
#
#    # randomly perturb their positions
#    num_points = k.sum()
#    x_in[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)
#    y_in[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)
#    z_in[k] += 0.2*dx*(2.0*np.random.random(num_points)-1.0)
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in); z_particles = np.copy(z_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#    z_particles = np.append(z_particles, z[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles, z_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh3D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    tot_vol = cells_info["volume"].sum()
#    assert np.abs(1.0 - tot_vol) < 1.0E-10
#
#def test_volume_x_3D():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a unit box. So
#    the total volume  is 1.0. Then perturb the particles
#    in a box of unit lenght 0.5. Create the tessellation
#    and compare the sum of all the particle volumes and
#    the total volume.
#    """
#    L = 10.0
#    n = 5
#
#    dx = L/n
#    qx = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
#
#    L = 1.0
#    n = 5
#
#    dq = L/n
#    q = (np.arange(n+6, dtype=np.float64) - 3)*dq + 0.5*dq
#
#    N = q.size
#    x = np.zeros(N**3)
#    y = np.zeros(N**3)
#    z = np.zeros(N**3)
#
#    part = 0
#    for i in xrange(N):
#        for j in xrange(N):
#            for k in xrange(N):
#                x[part] = qx[i]
#                y[part] = q[j]
#                z[part] = q[k]
#                part += 1
#
#    # find all particles inside the unit box 
#    indices = (((0. <= x) & (x <= 10.)) & ((0. <= y) & (y <= 1.)) & ((0. <= z) & (z <= 1.)))
#    x_in = x[indices]; y_in = y[indices]; z_in = z[indices]
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in); z_particles = np.copy(z_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#    z_particles = np.append(z_particles, z[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles, z_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh3D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    tot_vol = cells_info["volume"].sum()
#    assert np.abs(10.0 - tot_vol) < 1.0E-10
#
#def test_volume_y_3D():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a unit box. So
#    the total volume  is 1.0. Then perturb the particles
#    in a box of unit lenght 0.5. Create the tessellation
#    and compare the sum of all the particle volumes and
#    the total volume.
#    """
#    L = 10.0
#    n = 5
#
#    dy = L/n
#    qy = (np.arange(n+6, dtype=np.float64) - 3)*dy + 0.5*dy
#
#    L = 1.0
#    n = 5
#
#    dq = L/n
#    q = (np.arange(n+6, dtype=np.float64) - 3)*dq + 0.5*dq
#
#    N = q.size
#    x = np.zeros(N**3)
#    y = np.zeros(N**3)
#    z = np.zeros(N**3)
#
#    part = 0
#    for i in xrange(N):
#        for j in xrange(N):
#            for k in xrange(N):
#                x[part] = q[i]
#                y[part] = qy[j]
#                z[part] = q[k]
#                part += 1
#
#    # find all particles inside the unit box 
#    indices = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 10.)) & ((0. <= z) & (z <= 1.)))
#    x_in = x[indices]; y_in = y[indices]; z_in = z[indices]
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in); z_particles = np.copy(z_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#    z_particles = np.append(z_particles, z[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles, z_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh3D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    tot_vol = cells_info["volume"].sum()
#    assert np.abs(10.0 - tot_vol) < 1.0E-10
#
#def test_volume_y_3D():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a unit box. So
#    the total volume  is 1.0. Then perturb the particles
#    in a box of unit lenght 0.5. Create the tessellation
#    and compare the sum of all the particle volumes and
#    the total volume.
#    """
#    L = 10.0
#    n = 5
#
#    dz = L/n
#    qz = (np.arange(n+6, dtype=np.float64) - 3)*dz + 0.5*dz
#
#    L = 1.0
#    n = 5
#
#    dq = L/n
#    q = (np.arange(n+6, dtype=np.float64) - 3)*dq + 0.5*dq
#
#    N = q.size
#    x = np.zeros(N**3)
#    y = np.zeros(N**3)
#    z = np.zeros(N**3)
#
#    part = 0
#    for i in xrange(N):
#        for j in xrange(N):
#            for k in xrange(N):
#                x[part] = q[i]
#                y[part] = q[j]
#                z[part] = qz[k]
#                part += 1
#
#    # find all particles inside the unit box 
#    indices = (((0. <= x) & (x <= 1.)) & ((0. <= y) & (y <= 1.)) & ((0. <= z) & (z <= 10.)))
#    x_in = x[indices]; y_in = y[indices]; z_in = z[indices]
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in); z_particles = np.copy(z_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#    z_particles = np.append(z_particles, z[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles, z_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh3D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    tot_vol = cells_info["volume"].sum()
#    assert np.abs(10.0 - tot_vol) < 1.0E-10
#
#def test_volume_random_rectangel_3D():
#    """Test if particle volumes are created correctly.
#    First create a grid of particles in a rectangular
#    box [0,10]x[0,1], so the total volume is 10.0
#    """
#
#    xvals = -2*np.random.random(2) + 1
#    xmin = np.min(xvals)
#    xmax = np.max(xvals)
#
#    L = (xmax-xmin)  # size in x
#    n = 5            # number of points along dimension
#    dx = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    qx = xmin + (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
#
#    yvals = -2*np.random.random(2) + 1
#    ymin = np.min(yvals)
#    ymax = np.max(yvals)
#
#    L = (ymax-ymin)  # size in x
#    n = 5            # number of points along dimension
#    dy = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    qy = ymin + (np.arange(n+6, dtype=np.float64) - 3)*dy + 0.5*dy
#
#    zvals = -2*np.random.random(2) + 1
#    zmin = np.min(zvals)
#    zmax = np.max(zvals)
#
#    L = (zmax-zmin)  # size in x
#    n = 5            # number of points along dimension
#    dz = L/n
#
#    # add ghost 3 ghost particles to the sides for the tesselation
#    # wont suffer from edge boundaries
#    qz = zmin + (np.arange(n+6, dtype=np.float64) - 3)*dz + 0.5*dz
#
#    N = qx.size
#    x = np.zeros(N**3)
#    y = np.zeros(N**3)
#    z = np.zeros(N**3)
#
#    part = 0
#    for i in xrange(N):
#        for j in xrange(N):
#            for k in xrange(N):
#                x[part] = qx[i]
#                y[part] = qy[j]
#                z[part] = qz[k]
#                part += 1
#
#    # find all particles inside the box 
#    indices = (((xmin <= x) & (x <= xmax)) & ((ymin <= y) & (y <= ymax)) & ((zmin <= z) & (z <= zmax)))
#    x_in = x[indices]; y_in = y[indices]; z_in = z[indices]
#
#    # store real particles
#    x_particles = np.copy(x_in); y_particles = np.copy(y_in); z_particles = np.copy(z_in)
#    particles_index = {"real": np.arange(x_particles.size)}
#
#    # store ghost particles
#    x_particles = np.append(x_particles, x[~indices])
#    y_particles = np.append(y_particles, y[~indices])
#    z_particles = np.append(z_particles, z[~indices])
#
#    # store indices of ghost particles
#    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)
#
#    # particle list of real and ghost particles
#    particles = np.array([x_particles, y_particles, z_particles])
#
#    # generate voronoi mesh 
#    mesh = VoronoiMesh3D()
#    graphs = mesh.tessellate(particles)
#
#    # calculate voronoi volumes of all real particles 
#    cells_info, _ = mesh.cell_and_faces_info(particles, particles_index, graphs)
#
#    tot_vol = cells_info["volume"].sum()
#    assert np.abs((xmax-xmin)*(ymax-ymin)*(zmax-zmin) - tot_vol) < 1.0E-10
