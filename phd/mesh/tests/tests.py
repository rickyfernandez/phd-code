import unittest
import numpy as np

from phd.mesh.voronoi_mesh import VoronoiMesh2D
from utils.particle_tags import ParticleTAGS
from phd.containers.containers import ParticleContainer


class TestVoronoiMesh2dBox(unittest.TestCase):

    def setUp(self):

        L = 1.      # box size
        self.n = 50 # number of points
        self.dx = L/self.n

        # add ghost 3 ghost particles to the sides for the tesselation
        # wont suffer from edge boundaries
        x = (np.arange(self.n+6, dtype=np.float64) - 3)*self.dx + 0.5*self.dx

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
        type = self.particles["type"]
        type[:] = ParticleTAGS.Undefined
        self.particles.align_particles()


    def test_volume_2D(self):
        """Test particle volumes in square created correctly.
        Create grid of particles in a unit box, total volume
        is 1.0. Create tessellation and sum all particle
        volumes.
        """
        # generate voronoi mesh 
        mesh = VoronoiMesh2D(self.particles, None, None)
        mesh.tessellate()

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info()
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

        # generate voronoi mesh 
        mesh = VoronoiMesh2D(self.particles, None, None)
        mesh.tessellate()

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info()
        real_indices = self.particles["tag"] == ParticleTAGS.Real
        tot_vol = np.sum(self.particles["volume"][real_indices])

        self.assertAlmostEqual(tot_vol, 1.0)

    def test_find_boundary_particles(self):
        """Test if mesh correctly finds the ghost boundary particles.
        Since the particles are in a uniform lattice there should be
        4 times the number of particles in given dimension.
        """
        indices = np.arange(self.particles.get_number_of_particles())

        # generate voronoi mesh 
        mesh = VoronoiMesh2D(self.particles, None, None)
        mesh.tessellate()
        mesh.update_boundary_particles()

        # get sorted indices of boundary particles by mesh
        boundary = self.particles["type"] == ParticleTAGS.Boundary
        boundary_indices = np.sort(indices[boundary])

        self.assertEqual(boundary_indices.size, 4*self.n)


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
        mesh = VoronoiMesh2D(self.particles, None, None)
        mesh.tessellate()

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info()
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
        self.dy = L/n

        # add ghost 3 ghost particles to the sides for the tesselation
        # wont suffer from edge boundaries
        y = self.ymin + (np.arange(n+6, dtype=np.float64) - 3)*self.dy + 0.5*self.dy

        self.n = n

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
        type = self.particles["type"]
        type[:] = ParticleTAGS.Undefined
        self.particles.align_particles()


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
        mesh = VoronoiMesh2D(self.particles, None, None)
        mesh.tessellate()

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info()
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
        mesh = VoronoiMesh2D(self.particles, None, None)
        mesh.tessellate()

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info()
        real_indices = self.particles["tag"] == ParticleTAGS.Real
        tot_vol = np.sum(self.particles["volume"][real_indices])

        self.assertAlmostEqual(tot_vol, Lx*Ly)

    def test_find_boundary_particles(self):
        """Test if mesh correctly finds the ghost boundary particles.
        Since the particles are in a uniform lattice there should be
        4 times the number of particles in given dimension.
        """
        indices = np.arange(self.particles.get_number_of_particles())

        # generate voronoi mesh 
        mesh = VoronoiMesh2D(self.particles, None, None)
        mesh.tessellate()
        mesh.update_boundary_particles()

        # get sorted indices of boundary particles by mesh
        boundary = self.particles["type"] == ParticleTAGS.Boundary
        boundary_indices = np.sort(indices[boundary])

        self.assertEqual(boundary_indices.size, 4*self.n)

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
        mesh = VoronoiMesh2D(self.particles, None, None)
        mesh.tessellate()

        # calculate voronoi volumes of all real particles 
        mesh.compute_cell_info()
        real_indices = self.particles["tag"] == ParticleTAGS.Real

        xcom = self.particles["com-x"][real_indices];    ycom = self.particles["com-y"][real_indices]
        xp = self.particles["position-x"][real_indices]; yp = self.particles["position-y"][real_indices]

        for i in range(xp.size):
            self.assertAlmostEqual(xcom[i], xp[i])
            self.assertAlmostEqual(ycom[i], yp[i])
