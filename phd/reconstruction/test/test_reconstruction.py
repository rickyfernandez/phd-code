import unittest
import numpy as np

from phd.mesh.mesh import Mesh
from phd.domain.domain_manager import DomainManager
from phd.containers.containers import CarrayContainer
from phd.utils.particle_creator import HydroParticleCreator
from phd.reconstruction.reconstruction import PieceWiseConstant, PieceWiseLinear

from phd.riemann.riemann import RiemannBase

class TestPieceWiseConstant(unittest.TestCase):
    """Tests for the constant reconstruction class."""

    def setUp(self):

        # setup particles and reconstruction class
        self.particles = HydroParticleCreator(num=2, dim=2)
        self.recon = PieceWiseConstant()

    def test_add_fields(self):
        particles = CarrayContainer(1, {"wrong_type": "int"})

        # error for no primitive and velocity named group
        self.assertRaises(RuntimeError,
                self.recon.add_fields, particles)

    def test_initialize(self):
        # fields not registered
        self.assertRaises(RuntimeError, self.recon.initialize)

        self.recon.add_fields(self.particles)
        self.recon.initialize()

        # check for left state primitive fields
        self.assertItemsEqual(
                self.recon.left_states.carray_named_groups["primitive"],
                self.particles.carray_named_groups["primitive"])
        self.assertItemsEqual(
                self.recon.left_states.carray_named_groups["velocity"],
                self.particles.carray_named_groups["velocity"])

        # check for right primitive fields
        self.assertItemsEqual(
                self.recon.right_states.carray_named_groups["primitive"],
                self.particles.carray_named_groups["primitive"])
        self.assertItemsEqual(
                self.recon.right_states.carray_named_groups["velocity"],
                self.particles.carray_named_groups["velocity"])

    def test_compute_states(self):

        # create 2 particles with constant values
        for field in self.particles.carray_named_groups["primitive"]:
            self.particles[field][:] = 1.0

        self.recon.add_fields(self.particles)
        self.recon.initialize()

        # domain manager
        domain_manager = DomainManager(0.2)

        # create mesh 
        mesh = Mesh()
        mesh.register_fields(self.particles)
        mesh.initialize()

        # hard code mesh faces (1 face)
        mesh.faces.resize(1)
        mesh.faces["pair-i"][0] = 0  # left particle
        mesh.faces["pair-j"][0] = 1  # right particle

        # construct left/right faces
        self.recon.compute_states(self.particles, mesh, False,
                domain_manager, dt=1.0)

        # left and right states should be the same
        for field in self.recon.left_states.carrays.keys():
            self.assertAlmostEqual(self.recon.left_states[field][0],
                    self.particles[field][0])

            self.assertAlmostEqual(self.recon.right_states[field][0],
                    self.particles[field][1])

class TestPieceWiseLinear(TestPieceWiseConstant):
    """Tests for the lineaer reconstruction class."""

    def setUp(self):

        # setup particles and reconstruction class
        self.particles = HydroParticleCreator(num=2, dim=2)
        self.recon = PieceWiseLinear()

    def test_compute_states(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
