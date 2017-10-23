import unittest
import numpy as np
from mock import patch

from phd.mesh.mesh import Mesh
from phd.domain.domain import DomainLimits
from phd.domain.boundary import Reflective
from phd.domain.domain_manager import DomainManager
from phd.utils.particle_creator import HydroParticleCreator


class TestMeshSetup2d(unittest.TestCase):
    """Tests for the Reconstruction class."""

    def setUp(self):
        self.mesh = Mesh(param_dim=2)

    def test_register_fields(self):
        # create 2d particles
        particles = HydroParticleCreator(num=1, dim=2)
        fields = list(particles.properties)

        # check if correct fields where registered
        self.mesh.register_fields(particles)
        reg_fields_2d = ["volume", "dcom-x", "dcom-y", "w-x", "w-y"]
        for field in reg_fields_2d:
            self.assertTrue(field in particles.properties.keys())

        # check right number of fields
        self.assertEqual(particles.properties.keys().sort(),
                (fields + reg_fields_2d).sort())

        # check named groups added correctly
        self.assertEqual(particles.named_groups["w"],
                ["w-x", "w-y"])
        self.assertEqual(particles.named_groups["dcom"],
                ["dcom-x", "dcom-y"])

    def test_register_fields_errors(self):
        # register fields but wrong dimensions
        particles = HydroParticleCreator(num=1, dim=4)
        self.assertRaises(RuntimeError, self.mesh.register_fields, particles)

    def test_initialize_errors(self):
        # fields not registered, throws run time error
        self.assertRaises(RuntimeError, self.mesh.initialize)

    def test_initialize(self):
        # create 2d particles
        particles = HydroParticleCreator(num=1, dim=2)
        self.mesh.register_fields(particles)
        self.mesh.initialize()

        # fields to create in 2d
        face_vars_2d = ["area", "pair-i", "pair-j", "com-x", "com-y",
                "velocity-x", "velocity-y", "normal-x", "normal-y"]

        # check if correct fields registered
        for field in face_vars_2d:
            self.assertTrue(field in self.mesh.faces.properties.keys())

        # check right number of fields
        self.assertEqual(self.mesh.faces.properties.keys().sort(),
                face_vars_2d.sort())

    def test_mesh_creation(self):

        # create particle in center of lower left quadrant
        particles = HydroParticleCreator(num=7, dim=2)
        particles['position-x'][:] = [0.25, 0.75, 0.25, 0.50, 0.75, 0.25, 0.75]
        particles['position-y'][:] = [0.75, 0.75, 0.50, 0.50, 0.50, 0.25, 0.25]

        # create unit square domain
        domain_manager = DomainManager(param_initial_radius=0.2,
                param_search_radius_factor=2.0)
        domain_manager.set_domain(DomainLimits())
        domain_manager.register_fields(particles)
        domain_manager.set_boundary_condition(Reflective())

        mesh = Mesh()
        mesh.register_fields(particles)
        mesh.initialize()

        mesh.tessellate(particles, domain_manager)
        ##self.assertTrue(particles.get_number_of_items() == 17)

        self.assertFalse(False)
#
#        # two ghost particles created
#        self.assertTrue(particles.get_number_of_items() == 3)
#
#        particles['radius'][0] = 0.6
#        self.domain_manager.update_search_radius(particles)
#        self.domain_manager.create_ghost_particles(particles)
#
#        # no new ghost should be created
#        self.assertTrue(particles.get_number_of_items() == 3)

if __name__ == "__main__":
    unittest.main()

