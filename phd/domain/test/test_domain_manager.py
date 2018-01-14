import unittest
import numpy as np
from mock import patch

from phd.mesh.mesh import Mesh
from phd.domain.domain import DomainLimits
from phd.domain.boundary import Reflective, Periodic
from phd.domain.domain_manager import DomainManager
from phd.utils.particle_creator import HydroParticleCreator


class TestDomainManagerSetup(unittest.TestCase):
    """Tests for the Reconstruction class."""

    def setUp(self):

        # setup particles and reconstruction class
        self.domain_manager = DomainManager(param_initial_radius=0.25,
                param_search_radius_factor=2.0)

    def test_initialize(self):
        # boundary and domain not set
        self.assertRaises(RuntimeError, self.domain_manager.initialize)

    def test_check_initial_radius(self):

        # create particle in center of lower left quadrant
        particles = HydroParticleCreator(num=1, dim=2)
        particles['position-x'][0] = 0.125
        particles['position-y'][0] = 0.125

        # create unit square domain
        self.domain_manager.set_domain(DomainLimits())
        self.domain_manager.register_fields(particles)
        self.domain_manager.set_boundary_condition(Reflective())

        mesh = Mesh()
        mesh.register_fields(particles)
        mesh.initialize()

        # set infinte radius flag FIX add variable instead of magic number
        particles['radius'][0] = -1

        self.domain_manager.setup_for_ghost_creation(particles)
        self.assertTrue(particles['radius'][0] == 0.25)
        self.assertTrue(particles['old_radius'][0] == 0.25)

    def test_setup_for_ghost_creation_reflection(self):

        # create particle in center of lower left quadrant
        particles = HydroParticleCreator(num=1, dim=2)
        particles['position-x'][0] = 0.25
        particles['position-y'][0] = 0.25

        # create unit square domain
        self.domain_manager.set_domain(DomainLimits())
        self.domain_manager.register_fields(particles)
        self.domain_manager.set_boundary_condition(Reflective())

        mesh = Mesh()
        mesh.register_fields(particles)
        mesh.initialize()

        # set infinte radius flag FIX add variable instead of magic number
        particles['radius'][0] = -1

        self.domain_manager.setup_for_ghost_creation(particles)
        self.domain_manager.create_ghost_particles(particles)

        # two ghost particles created
        self.assertTrue(particles.get_carray_size() == 3)

        # two ghost particles should of been created
        # first particle reflected across x-min
        self.assertEqual(particles['position-x'][1], -0.25)
        self.assertEqual(particles['position-y'][1],  0.25)

        # second particle reflected across y-min
        self.assertEqual(particles['position-x'][2],  0.25)
        self.assertEqual(particles['position-y'][2], -0.25)

        # should have particle in flagged buffer
        self.assertFalse(self.domain_manager.ghost_complete())

        # update search radius to remove particle from being flagged
        particles['radius'][1:] = 0.3
        self.domain_manager.update_search_radius(particles)
        self.assertTrue(self.domain_manager.ghost_complete())

    def test_second_pass_reflection_serial(self):

        # create particle in center of lower left quadrant
        particles = HydroParticleCreator(num=1, dim=2)
        particles['position-x'][0] = 0.25
        particles['position-y'][0] = 0.25

        # create unit square domain
        self.domain_manager.set_domain(DomainLimits())
        self.domain_manager.register_fields(particles)
        self.domain_manager.set_boundary_condition(Reflective())

        mesh = Mesh()
        mesh.register_fields(particles)
        mesh.initialize()

        # set infinte radius flag FIX add variable instead of magic number
        particles['radius'][0] = -1

        self.domain_manager.setup_for_ghost_creation(particles)
        self.domain_manager.create_ghost_particles(particles)

        # two ghost particles created
        self.assertTrue(particles.get_carray_size() == 3)

        particles['radius'][0] = 0.6
        self.domain_manager.update_search_radius(particles)
        self.domain_manager.create_ghost_particles(particles)

        # no new ghost should be created
        self.assertTrue(particles.get_carray_size() == 3)


#    def test_setup_for_ghost_creation_periodic(self):
#
#        # create particle in center of lower left quadrant
#        particles = HydroParticleCreator(num=1, dim=2)
#        particles['position-x'][0] = 0.25
#        particles['position-y'][0] = 0.25
#
#        # create unit square domain
#        self.domain_manager.set_domain(DomainLimits())
#        self.domain_manager.register_fields(particles)
#        self.domain_manager.set_boundary_condition(Periodic())
#
#        mesh = Mesh()
#        mesh.register_fields(particles)
#        mesh.initialize()
#
#        # set infinte radius flag FIX add variable instead of magic number
#        particles['radius'][0] = -1
#
#        self.domain_manager.setup_for_ghost_creation(particles)
#        self.domain_manager.create_ghost_particles(particles)
#
#        # three ghost particles shoulb be created
#        self.assertTrue(particles.get_carray_size() == 4)
#
#        # first ghost particle, see boundary documentation
#        # for shift pattern
#        self.assertEqual(particles['position-x'][1], 1.25)
#        self.assertEqual(particles['position-y'][1], 0.25)
#
#        # second particle
#        self.assertEqual(particles['position-x'][2], 0.25)
#        self.assertEqual(particles['position-y'][2], 1.25)
#
#        # third particle
#        self.assertEqual(particles['position-x'][3], 1.25)
#        self.assertEqual(particles['position-y'][3], 1.25)
#
#        # should have particles in flagged buffer
#        self.assertFalse(self.domain_manager.ghost_complete())
#
#        # update search radius to remove particle from being flagged
#        particles['radius'][1:] = 0.3
#        self.domain_manager.update_search_radius(particles)
#        self.assertTrue(self.domain_manager.ghost_complete())

if __name__ == "__main__":
    unittest.main()

