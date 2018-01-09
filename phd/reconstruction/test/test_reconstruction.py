import unittest
import numpy as np

from phd.mesh.mesh import Mesh
from phd.domain.domain_manager import DomainManager
from phd.containers.containers import CarrayContainer
from phd.utils.particle_creator import HydroParticleCreator
from phd.reconstruction.reconstruction import PieceWiseConstant

from phd.riemann.riemann import RiemannBase

class TestPieceWiseConstantSetup(unittest.TestCase):
    """Tests for the Reconstruction class."""

    def setUp(self):

        # setup particles and reconstruction class
        self.particles = HydroParticleCreator(num=1, dim=2)
        self.reconstruction = PieceWiseConstant()

    def test_set_fields_for_reconstruction_error(self):
        particles = CarrayContainer(1, {"wrong_type": "int"})
        particles.carray_named_groups["primitive"] = []
        particles.carray_named_groups["primitive"].append("wrong_type")

        self.assertRaises(RuntimeError,
                self.reconstruction.set_fields_for_reconstruction, particles)

    def test_initialize_errors(self):
        # fields for reconstruction not registered
        self.assertRaises(RuntimeError, self.reconstruction.initialize)

    def test_set_fields_for_reconstruction(self):

       self.reconstruction.set_fields_for_reconstruction(self.particles)

       # check for primitive fields, these are the fields that
       # will be reconstructed
       self.assertEqual(
               self.reconstruction.reconstruct_field_groups["primitive"],
               self.particles.carray_named_groups["primitive"])

       # check for velocity fields
       self.assertEqual(
               self.reconstruction.reconstruct_field_groups["velocity"],
               self.particles.carray_named_groups["velocity"])

       # check if fields type are correct
       for field in self.reconstruction.reconstruct_fields.keys():
           self.assertEqual(
                   self.reconstruction.reconstruct_fields[field],
                   self.particles.carray_info[field])

    def test_initialize(self):
       self.reconstruction.set_fields_for_reconstruction(self.particles)

       # create left/right states containers
       self.reconstruction.initialize()

       # check if left states have correct fields
       self.assertEqual(
               self.reconstruction.left_states.carray_named_groups["primitive"],
               self.particles.carray_named_groups["primitive"])

       # check if rigth states have correct fields
       self.assertEqual(
               self.reconstruction.right_states.carray_named_groups["primitive"],
               self.particles.carray_named_groups["primitive"])

       # check if left states have correct velocity fields
       self.assertEqual(
               self.reconstruction.left_states.carray_named_groups["velocity"],
               self.particles.carray_named_groups["velocity"])

       # check if rigth states have correct fields
       self.assertEqual(
               self.reconstruction.right_states.carray_named_groups["velocity"],
               self.particles.carray_named_groups["velocity"])

class TestPieceWiseConstantComputeStates(unittest.TestCase):
    def setUp(self):

        # create 2 particles with constant values
        self.particles = HydroParticleCreator(num=2, dim=2)
        for field in self.particles.carray_named_groups["primitive"]:
            self.particles[field][:] = 1.0

        # reconstruction class
        self.reconstruction = PieceWiseConstant()
        self.reconstruction.set_fields_for_reconstruction(
                self.particles)
        self.reconstruction.initialize()

    def test_compute_states(self):
        """Test reconstruction to left and right of face."""

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
        self.reconstruction.compute_states(self.particles, mesh, False,
                domain_manager, dt=1.0)
        self.assertTrue(True)

        # left and right states should be the same
        for field in self.reconstruction.left_states.properties.keys():
            self.assertAlmostEqual(self.reconstruction.left_states[field][0],
                    self.particles[field][0])
            self.assertAlmostEqual(self.reconstruction.right_states[field][0],
                    self.particles[field][1])

if __name__ == "__main__":
    unittest.main()

# patch rieman class
#self.riemann_patch = patch('phd.riemann.riemann.RiemannBase', spec=True)
#self.riemann = self.riemann_patch.start()


#self.reconstruction = PieceWiseConstant()

#def tearDown(self):
#    self.riemann_patch.stop()
#    self.domain_manager_patch.stop()
# patch domain manager class
#self.riemann_patch = patch('phd.riemann.riemann.RiemannBase', spec=RiemannBase)
#self.riemann = self.riemann_patch.start()

# patch equation of state
#self.domain_manager_patch = patch('phd.domain.domain_manager.DomainManager',
#        spec=DomainManager)
#self.domain_manager = self.domain_manager_patch.start()
