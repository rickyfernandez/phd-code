import unittest
import numpy as np

from phd.particles.particle_array import ParticleArray

def check_array(x, y):
    """Check if two array are equal with an absolute tolerance of
    1e-16."""
    return np.allclose(x, y , atol=1e-16, rtol=0)


class TestParticleArray(unittest.TestCase):
    """Tests for the ParticleArray class."""
    def test_constructor(self):
        """Test the constructor."""
        pc = ParticleArray(10)

        self.assertEqual(pc.get_number_of_particles(), 10)

        expected = ['position-x', 'position-y', 'mass', 'momentum-x',
                'momentum-y', 'energy', 'density', 'velocity-x',
                'velocity-y', 'pressure', 'key', 'tag', 'process']

        self.assertItemsEqual(pc.properties.keys(), expected)

        expected = ['mass', 'momentum-x', 'momentum-y', 'energy',
                'density', 'velocity-x', 'velocity-y', 'pressure']
        self.assertItemsEqual(pc.field_names, expected)

        for field_name in pc.field_names:
            self.assertEqual(pc[field_name].size, 10)

    def test_get_number_of_particles(self):
        """
        Tests the get_number_of_particles of particles.
        """
        p = ParticleArray(4)
        self.assertEqual(p.get_number_of_particles(), 4)

    def test_remove_particles(self):
        """"Test the discard ghost and export particles function"""
        x = [5.0, 2.0, 1.0, 4.0]
        y = [2.0, 6.0, 3.0, 1.0]
        m = [1.0, 2.0, 3.0, 4.0]
        u = [0.0, 1.0, 2.0, 3.0]
        v = [1.0, 1.0, 1.0, 1.0]
        e = [1.0, 1.0, 1.0, 1.0]

        pa = ParticleArray(4)

        xpos = pa['position-x']
        ypos = pa['position-y']
        mass = pa['mass']
        momx = pa['momentum-x']
        momy = pa['momentum-y']
        ener = pa['energy']

        xpos[:] = x
        ypos[:] = y
        mass[:] = m
        momx[:] = u
        momy[:] = v
        ener[:] = e

        remove_arr = np.array([0, 1], dtype=np.int)
        pa.remove_particles(remove_arr)

        self.assertEqual(pa.get_number_of_particles(), 2)
        self.assertEqual(check_array(pa['position-x'], [1.0, 4.0]), True)
        self.assertEqual(check_array(pa['position-y'], [3.0, 1.0]), True)
        self.assertEqual(check_array(pa['mass'], [3.0, 4.0]), True)
        self.assertEqual(check_array(pa['momentum-x'], [2.0, 3.0]), True)
        self.assertEqual(check_array(pa['momentum-y'], [1.0, 1.0]), True)
        self.assertEqual(check_array(pa['energy'], [1.0, 1.0]), True)

        # now try invalid operations to make sure errors are raised
        remove = np.arange(10, dtype=np.int)
        self.assertRaises(ValueError, pa.remove_particles, remove)

        remove = np.array([2], dtype=np.int)
        pa.remove_particles(remove)

        # make sure no change has occured
        self.assertEqual(pa.get_number_of_particles(), 2)
        self.assertEqual(check_array(pa['position-x'], [1.0, 4.0]), True)
        self.assertEqual(check_array(pa['position-y'], [3.0, 1.0]), True)
        self.assertEqual(check_array(pa['mass'], [3.0, 4.0]), True)
        self.assertEqual(check_array(pa['momentum-x'], [2.0, 3.0]), True)
        self.assertEqual(check_array(pa['momentum-y'], [1.0, 1.0]), True)
        self.assertEqual(check_array(pa['energy'], [1.0, 1.0]), True)

    def test_remove_tagged_particles(self):
        """
        Tests the remove_tagged_particles function.
        """

        pa = ParticleArray(4)
        pa['position-x'][:] = [1, 2, 3, 4.]
        pa['position-y'][:] = [0., 1., 2., 3.]
        pa['mass'][:] = [1., 1., 1., 1.]
        pa['tag'][:] = [1, 1, 1, 0]

        pa.remove_tagged_particles(0)

        self.assertEqual(pa.get_number_of_particles(), 3)
        self.assertEqual(check_array(pa['position-x'], [1, 2, 3.]), True)
        self.assertEqual(check_array(pa['position-y'], [0., 1, 2]), True)
        self.assertEqual(check_array(pa['mass'], [1., 1., 1.]), True)
        self.assertEqual(check_array(pa['tag'], [1, 1, 1]), True)

    def test_align_particles(self):
        """
        Tests the align particles function.
        """
        pa = ParticleArray(10)
        pa['position-x'][:] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pa['position-y'][:] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        pa['tag'][:] = [0, 0, 1, 1, 1, 0, 4, 0, 1, 5]

        pa.align_particles()
        self.assertEqual(check_array(pa['position-x'],
                                     [1, 2, 6, 8, 5, 3, 7, 4, 9, 10]), True)
        self.assertEqual(check_array(pa['position-y'],
                                     [10, 9, 5, 3, 6, 8, 4, 7, 2, 1]), True)

        # should do nothing
        pa['tag'][:] = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        pa.align_particles()
        self.assertEqual(check_array(pa['position-x'],
                                     [1, 2, 6, 8, 5, 3, 7, 4, 9, 10]), True)
        self.assertEqual(check_array(pa['position-y'],
                                     [10, 9, 5, 3, 6, 8, 4, 7, 2, 1]), True)

    def test_extend(self):
        """
        Tests the extend function.
        """
        pa = ParticleArray()
        self.assertEqual(pa.get_number_of_particles(), 0)
        print "num:", pa.get_number_of_particles()

        pa.extend(100)

        print "num:", pa.get_number_of_particles()
        self.assertEqual(pa.get_number_of_particles(), 100)
        for field in pa.properties.itervalues():
            self.assertEqual(field.length, 100)

    def test_extract_particles(self):
        """
        Tests the extract particles function.
        """
        pa = ParticleArray(10)
        pa['position-x'][:] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pa['position-y'][:] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        pa['tag'][:] = [0, 0, 1, 1, 1, 0, 4, 0, 1, 5]

        indices = np.array([5, 1, 7, 3, 9])
        pa2 = pa.extract_particles(indices)

        self.assertEqual(check_array(pa2['position-x'],
                                     [6, 2, 8, 4, 10]), True)
        self.assertEqual(check_array(pa2['position-y'],
                                     [5, 9, 3, 7, 1]), True)
        self.assertEqual(check_array(pa2['tag'],
                                     [0, 0, 0, 1, 5]), True)
