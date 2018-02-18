import unittest
import numpy as np

from phd.utils.particle_tags import ParticleTAGS

from phd.mesh.mesh import Mesh
from phd.domain.domain import DomainLimits
from phd.domain.boundary import Reflective
from phd.domain.domain_manager import DomainManager
from phd.utils.particle_creator import HydroParticleCreator


class TestMeshSetup2d(unittest.TestCase):
    """Tests for the Reconstruction class."""

    def setUp(self):
        self.dim = 2
        self.mesh = Mesh()

    def test_register_fields(self):
        particles = HydroParticleCreator(num=1, dim=self.dim)
        fields = list(particles.carrays)

        # check if correct fields where registered
        self.mesh.register_fields(particles)
        reg_fields_2d = ["volume", "dcom-x", "dcom-y", "w-x", "w-y"]
        for field in reg_fields_2d:
            self.assertTrue(field in particles.carrays.keys())

        # check right number of fields
        self.assertEqual(particles.carrays.keys().sort(),
                (fields + reg_fields_2d).sort())

        # check named groups added correctly
        self.assertEqual(particles.carray_named_groups["w"],
                ["w-x", "w-y"])
        self.assertEqual(particles.carray_named_groups["dcom"],
                ["dcom-x", "dcom-y"])

    def test_initialize_errors(self):
        # fields not registered, throws run time error
        self.assertRaises(RuntimeError, self.mesh.initialize)

    def test_initialize(self):
        # create 2d particles
        particles = HydroParticleCreator(num=1, dim=self.dim)
        self.mesh.register_fields(particles)
        self.mesh.initialize()

        # fields to create in 2d
        face_vars_2d = ["area", "pair-i", "pair-j", "com-x", "com-y",
                "velocity-x", "velocity-y", "normal-x", "normal-y"]

        # check if correct fields registered
        for field in face_vars_2d:
            self.assertTrue(field in self.mesh.faces.carrays.keys())

        # check right number of fields
        self.assertEqual(self.mesh.faces.carrays.keys().sort(),
                face_vars_2d.sort())


class TestMeshSetup3d(TestMeshSetup2d):
    def setUp(self):
        self.dim = 3
        self.mesh = Mesh()

    def test_register_fields(self):
        particles = HydroParticleCreator(num=1, dim=self.dim)
        fields = list(particles.carrays)

        # check if correct fields where registered
        self.mesh.register_fields(particles)
        reg_fields_3d = ["volume", "dcom-x", "dcom-y", "dcom-z", "w-x", "w-y", "w-z"]
        for field in reg_fields_3d:
            self.assertTrue(field in particles.carrays.keys())

        # check right number of fields
        self.assertEqual(particles.carrays.keys().sort(),
                (fields + reg_fields_3d).sort())

        # check named groups added correctly
        self.assertEqual(particles.carray_named_groups["w"],
                ["w-x", "w-y", "w-z"])
        self.assertEqual(particles.carray_named_groups["dcom"],
                ["dcom-x", "dcom-y", "dcom-z"])

    def test_initialize(self):
        particles = HydroParticleCreator(num=1, dim=self.dim)
        self.mesh.register_fields(particles)
        self.mesh.initialize()

        # fields to create in 2d
        face_vars_3d = ["area", "pair-i", "pair-j",
                "com-x", "com-y", "com-z",
                "velocity-x", "velocity-y", "velocity-z",
                "normal-x", "normal-y", "normal-z"]

        # check if correct fields registered
        for field in face_vars_3d:
            self.assertTrue(field in self.mesh.faces.carrays.keys())

        # check right number of fields
        self.assertEqual(self.mesh.faces.carrays.keys().sort(),
                face_vars_3d.sort())

class TestMesh2dUniformBox(unittest.TestCase):

    def setUp(self):
        n = 100
        self.particles = HydroParticleCreator(num=n, dim=2)

        # create uniform random particles in a unit box
        np.random.seed(0)
        self.particles["position-x"][:] = np.random.uniform(size=n)
        self.particles["position-y"][:] = np.random.uniform(size=n)

        # create unit square domain, reflective boundary condition
        minx = np.array([0., 0.])
        maxx = np.array([1., 1.])
        self.domain_manager = DomainManager(initial_radius=0.1,
                search_radius_factor=1.25)
        self.domain_manager.set_domain_limits(DomainLimits(minx, maxx))
        self.domain_manager.register_fields(self.particles)
        self.domain_manager.set_boundary_condition(Reflective())
        self.domain_manager.initialize()

        self.mesh = Mesh()
        self.mesh.register_fields(self.particles)
        self.mesh.initialize()

    def test_volume(self):
        """
        Test if particle volumes in a square are created correctly.
        Create grid of particles in a unit box, total volume is 1.0.
        """
        # generate voronoi mesh 
        self.mesh.build_geometry(self.particles, self.domain_manager)

        # calculate voronoi volumes of all real particles 
        real_indices = self.particles["tag"] == ParticleTAGS.Real
        tot_vol = np.sum(self.particles["volume"][real_indices])

        # total mass should be equal to the volume of the box
        self.assertAlmostEqual(tot_vol, 1.0)

class TestMesh3dUniformBox(TestMesh2dUniformBox):

    def setUp(self):
        n = 50**3
        self.particles = HydroParticleCreator(num=n, dim=3)

        # create uniform random particles in a unit box
        np.random.seed(0)
        self.particles["position-x"][:] = np.random.uniform(size=n)
        self.particles["position-y"][:] = np.random.uniform(size=n)
        self.particles["position-z"][:] = np.random.uniform(size=n)

        # create unit square domain, reflective boundary condition
        minx = np.array([0., 0., 0.])
        maxx = np.array([1., 1., 1.])
        self.domain_manager = DomainManager(initial_radius=0.1,
                search_radius_factor=1.25)
        self.domain_manager.set_domain_limits(DomainLimits(minx, maxx, dim=3))
        self.domain_manager.register_fields(self.particles)
        self.domain_manager.set_boundary_condition(Reflective())
        self.domain_manager.initialize()

        self.mesh = Mesh()
        self.mesh.register_fields(self.particles)
        self.mesh.initialize()

class TestMesh2dLatticeBox(unittest.TestCase):
    def setUp(self):
        nx = ny = 10
        n = nx*nx

        self.particles = HydroParticleCreator(num=n, dim=2)

        # create lattice particles in a unit box
        L = 1.
        dx = L/nx; dy = L/ny
        self.volume = dx*dy

        part = 0
        for i in range(nx):
            for j in range(ny):
                self.particles['position-x'][part] = (i+0.5)*dx
                self.particles['position-y'][part] = (j+0.5)*dy
                part += 1

        # create unit square domain, reflective boundary condition
        minx = np.array([0., 0.])
        maxx = np.array([1., 1.])
        self.domain_manager = DomainManager(initial_radius=0.1,
                search_radius_factor=1.25)
        self.domain_manager.set_domain_limits(DomainLimits(minx, maxx))
        self.domain_manager.register_fields(self.particles)
        self.domain_manager.set_boundary_condition(Reflective())
        self.domain_manager.initialize()

        self.mesh = Mesh()
        self.mesh.register_fields(self.particles)
        self.mesh.initialize()

    def test_volume(self):
        """
        Test if particle volumes in a square are created correctly.
        Create grid of particles in a unit box, total volume is 1.0.
        """
        # generate voronoi mesh 
        self.mesh.build_geometry(self.particles, self.domain_manager)

        # sum voronoi volumes of all real particles 
        real_indices = self.particles["tag"] == ParticleTAGS.Real
        tot_vol = np.sum(self.particles["volume"][real_indices])

        # total mass should be equal to the volume of the box
        self.assertAlmostEqual(tot_vol, 1.0)

    def test_center_of_mass(self):
        """
        Test if particle center of mass positions are created correctly.
        Particles are placed in a uniform lattice. Therefore the center
        of mass is the same as particle positions.
        """
        # generate voronoi mesh 
        self.mesh.build_geometry(self.particles, self.domain_manager)
        dim = len(self.particles.carray_named_groups["position"])
        axis = "xyz"[:dim]

        # check if particle position is the same as center of mass
        for i in range(self.particles.get_carray_size()):
            if self.particles["tag"][i] == ParticleTAGS.Real:
                for ax in axis:
                    self.assertAlmostEqual(0., self.particles["dcom-"+ax][i])

    def test_particle_volume(self):
        """
        Test if particle center of mass positions are created correctly.
        Particles are placed in a uniform lattice. Therefore the center
        of mass is the same as particle positions.
        """
        # generate voronoi mesh 
        self.mesh.build_geometry(self.particles, self.domain_manager)

        # check if particle position is the same as center of mass
        for i in range(self.particles.get_carray_size()):
            if self.particles["tag"][i] == ParticleTAGS.Real:
                self.assertAlmostEqual(self.volume, self.particles["volume"][i])


class TestMesh3dLattice(TestMesh2dLatticeBox):
    def setUp(self):
        nx = ny = nz = 10
        n = nx*nx*nz

        self.particles = HydroParticleCreator(num=n, dim=3)

        # create lattice particles in a unit box
        L = 1.
        dx = L/nx; dy = L/ny; dz = L/nz
        self.volume = dx*dy*dz

        part = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    self.particles['position-x'][part] = (i+0.5)*dx
                    self.particles['position-y'][part] = (j+0.5)*dy
                    self.particles['position-z'][part] = (k+0.5)*dz
                    part += 1

        # create unit square domain, reflective boundary condition
        minx = np.array([0., 0., 0.])
        maxx = np.array([1., 1., 1.])
        self.domain_manager = DomainManager(initial_radius=0.1,
                search_radius_factor=1.25)
        self.domain_manager.set_domain_limits(DomainLimits(minx, maxx, dim=3))
        self.domain_manager.register_fields(self.particles)
        self.domain_manager.set_boundary_condition(Reflective())
        self.domain_manager.initialize()

        self.mesh = Mesh()
        self.mesh.register_fields(self.particles)
        self.mesh.initialize()

if __name__ == "__main__":
    unittest.main()

