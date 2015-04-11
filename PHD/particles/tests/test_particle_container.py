import unittest
import numpy as np

from utils.carray import DoubleArray, IntArray
from particles.particle_container import ParticleContainer

def check_array(x, y):
    """Check if two array are equal with an absolute tolerance of
    1e-16."""
    return np.allclose(x, y , atol=1e-16, rtol=0)


class TestParticleContainer(unittest.TestCase):
    """Tests for the ParticleContainer class."""
    def test_constructor(self):
        """Test the constructor."""
        pc = ParticleContainer(10)

        self.assertEqual(pc.num_real_particles, 10)

        expected = ['position-x', 'position-y', 'mass', 'momentum-x',
                'momentum-y', 'energy', 'tag', 'key']
        self.assertItemsEqual(pc.properties.keys(), expected)

        expected = ['mass', 'momentum-x', 'momentum-y', 'energy']
        self.assertItemsEqual(pc.field_names, expected)

        for field_name in pc.field_names:
            self.assertEqual(pc[field_name].size, 10)

    def test_discard_ghost_and_export_particles(self):
        """"Test the discard ghost and export particles function"""
        x = [5.0, 2.0, 1.0, 4.0]
        y = [2.0, 6.0, 3.0, 1.0]
        m = [1.0, 2.0, 3.0, 4.0]
        u = [0.0, 1.0, 2.0, 3.0]
        v = [1.0, 1.0, 1.0, 1.0]
        e = [1.0, 1.0, 1.0, 1.0]

        pc = ParticleContainer(4)

        xpos = pc['position-x']
        ypos = pc['position-y']
        mass = pc['mass']
        momx = pc['momentum-x']
        momy = pc['momentum-y']
        ener = pc['energy']

        xpos[:] = x
        ypos[:] = y
        mass[:] = m
        momx[:] = u
        momy[:] = v
        ener[:] = e

        remove = np.array([0, 1], dtype=np.int)
        pc.discard_ghost_and_export_particles(remove)

        self.assertEqual(pc.num_real_particles, 2)
        self.assertEqual(check_array(pc['position-x'], [1.0, 4.0]), True)
        self.assertEqual(check_array(pc['position-y'], [3.0, 1.0]), True)
        self.assertEqual(check_array(pc['mass'], [3.0, 4.0]), True)
        self.assertEqual(check_array(pc['momentum-x'], [2.0, 3.0]), True)
        self.assertEqual(check_array(pc['momentum-y'], [1.0, 1.0]), True)
        self.assertEqual(check_array(pc['energy'], [1.0, 1.0]), True)

        # now try invalid operations to make sure errors are raised
        remove = np.arange(10, dtype=np.int)
        self.assertRaises(ValueError, pc.discard_ghost_and_export_particles, remove)

        remove = np.array([2], dtype=np.int)
        pc.discard_ghost_and_export_particles(remove)

        # make sure no change has occured
        self.assertEqual(pc.num_real_particles, 2)
        self.assertEqual(check_array(pc['position-x'], [1.0, 4.0]), True)
        self.assertEqual(check_array(pc['position-y'], [3.0, 1.0]), True)
        self.assertEqual(check_array(pc['mass'], [3.0, 4.0]), True)
        self.assertEqual(check_array(pc['momentum-x'], [2.0, 3.0]), True)
        self.assertEqual(check_array(pc['momentum-y'], [1.0, 1.0]), True)
        self.assertEqual(check_array(pc['energy'], [1.0, 1.0]), True)
