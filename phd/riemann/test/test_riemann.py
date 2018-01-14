import unittest
import numpy as np


from phd.mesh.mesh import Mesh
from phd.riemann.riemann import HLL
from phd.equation_state.equation_state import IdealGas
from phd.utils.particle_creator import HydroParticleCreator
from phd.reconstruction.reconstruction import PieceWiseConstant

class TestHLLSetup(unittest.TestCase):
    """Tests for the Reconstruction class."""

    def setUp(self):

       # setup particles and riemann class
       self.particles = HydroParticleCreator(num=1, dim=2)
       self.riemann = HLL()

    def test_set_fields_for_flux(self):

       self.riemann.fields_to_add(self.particles)

        # check if reconstruction has fields registered
       self.assertTrue(self.riemann.registered_fields)

       # check for conservative fields, these are the fields that
       # the riemann solver will use 
       self.assertEqual(
               self.riemann.flux_field_groups["conservative"],
               self.particles.carray_named_groups["conservative"])

       # check for momentum fields
       self.assertEqual(
               self.riemann.flux_field_groups["momentum"],
               self.particles.carray_named_groups["momentum"])

       # check if fields type are correct
       for field in self.riemann.flux_fields.keys():
           self.assertEqual(
                   self.riemann.flux_fields[field],
                   self.particles.carrays_dtypes[field])

    def test_initialize(self):
       self.riemann.fields_to_add(self.particles)

       # create left/right states containers
       self.riemann.initialize()

       # check if flux states have correct fields
       self.assertEqual(
               self.riemann.fluxes.carray_named_groups["conservative"],
               self.particles.carray_named_groups["conservative"])

       # check if flux have correct momentum fields
       self.assertEqual(
               self.riemann.fluxes.carray_named_groups["momentum"],
               self.particles.carray_named_groups["momentum"])


class TestHLLFlux(unittest.TestCase):
    """Tests for the Reconstruction class."""
    def setUp(self):
        """Test reconstruction to left and right of face."""

        # create 2 particles with constant values
        self.particles = HydroParticleCreator(num=2, dim=2)

        # mesh class
        self.mesh = Mesh()
        self.mesh.register_fields(self.particles)
        self.mesh.initialize()

        # equation of state class
        self.eos = IdealGas(gamma=1.4)

        # reconstruction class
        self.reconstruction = PieceWiseConstant()
        self.reconstruction.set_fields_for_reconstruction(
                self.particles)
        self.reconstruction.initialize()

        # riemann class
        self.riemann = HLL(boost=False)
        self.riemann.fields_to_add(self.particles)
        self.riemann.initialize()

    def test_left_state(self):

        lt = self.reconstruction.left_states
        rt = self.reconstruction.right_states
        lt.resize(1); rt.resize(1)

        # both particles have the same value
        lt["density"][:]    = 1.0; rt["density"][:]    = 1.0
        lt["velocity-x"][:] = 2.0; rt["velocity-x"][:] = 2.0
        lt["velocity-y"][:] = 0.0; rt["velocity-y"][:] = 0.0
        lt["pressure"][:]   = 0.4; rt["pressure"][:]   = 0.4

        faces = self.mesh.faces
        faces.resize(1)

        # the face is perpendicular to the x-axis and has velocity zero
        faces["normal-x"][0]   = 1; faces["normal-y"][0]   = 0
        faces["velocity-x"][0] = 0; faces["velocity-y"][0] = 0

        ans = {"mass": 2.0, "momentum-x": 4.4, "momentum-y": 0.0, "energy": 6.8}
        self.riemann.compute_fluxes(self.particles, self.mesh,
                self.reconstruction, self.eos)

        for field in self.riemann.fluxes.carrays.keys():
            self.assertAlmostEqual(self.riemann.fluxes[field][0], ans[field])

        # reverse fluid flow
        lt["velocity-x"][:] = -2.; rt["velocity-x"][:] = -2.
        ans = {"mass": -2.0, "momentum-x": 4.4, "momentum-y": 0.0, "energy": -6.8}
        self.riemann.compute_fluxes(self.particles, self.mesh,
                self.reconstruction, self.eos)

        # left and right states should be the same
        for field in self.riemann.fluxes.carrays.keys():
            self.assertAlmostEqual(self.riemann.fluxes[field][0], ans[field])

if __name__ == "__main__":
    unittest.main()
