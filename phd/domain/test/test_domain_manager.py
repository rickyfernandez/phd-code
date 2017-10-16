import unittest
import numpy as np
from mock import patch

from phd.mesh.mesh import Mesh
from phd.domain.domain import DomainLimits
from phd.domain.boundary import Reflective
from phd.domain.domain_manager import DomainManager
from phd.utils.particle_creator import HydroParticleCreator


class TestDomainManagerSetup(unittest.TestCase):
    """Tests for the Reconstruction class."""

    def setUp(self):

        # setup particles and reconstruction class
        self.domain_manager = DomainManager(param_initial_radius=0.25,
                param_box_fraction=0.5, param_search_radius_factor=2.0)

    def test_initialize(self):
        # boundary and domain not set
        self.assertRaises(RuntimeError, self.domain_manager.initialize)

    def test_create_ghost_particles(self):

        # create unit square domain
        self.domain_manager.set_domain(DomainLimits())
        self.domain_manager.set_boundary_condition(Reflective())

        # create particle in center of lower left quadrant
        particles = HydroParticleCreator(num=1, dim=2)
        particles['position-x'][0] = 0.125
        particles['position-y'][0] = 0.125

        mesh = Mesh()
        mesh.register_fields(particles)
        mesh.initialize()

        # set infinte radius flag FIX add variable instead of magic number
        particles['radius'][0] = -1

        self.domain_manager.setup_for_ghost_creation(particles)
        self.assertTrue(particles['radius'][0] == 0.25)

if __name__ == "__main__":
    unittest.main()

