import unittest
import numpy as np

from phd.containers.containers import CarrayContainer, ParticleContainer

def check_array(x, y):
    """Check if two array are equal with an absolute tolerance of
    1e-16."""
    return np.allclose(x, y , atol=1e-16, rtol=0)


class TestParticleArray(unittest.TestCase):
    """Tests for the ParticleArray class."""
    def test_constructor(self):
        """Test the constructor."""
        pc = ParticleContainer(10)

        self.assertEqual(pc.get_number_of_particles(), 10)

        expected = ['position-x', 'position-y', 'mass', 'momentum-x',
                'momentum-y', 'energy', 'density', 'velocity-x',
                'velocity-y', 'pressure', 'key', 'tag', 'process']

        self.assertItemsEqual(pc.properties.keys(), expected)

        for field_name in pc.properties.keys():
            self.assertEqual(pc[field_name].size, 10)

    def test_constructor_dict(self):
        """Test the constructor using dict."""

        flux_vars = {
                "mass": "double",
                "momentum-x": "double",
                "momentum-y": "double",
                "energy": "double",
                }
        cc = CarrayContainer(var_dict=flux_vars)

        self.assertEqual(cc.get_number_of_items(), 0)
        self.assertItemsEqual(cc.properties.keys(), flux_vars.keys())

        for field_name in cc.properties.keys():
            self.assertEqual(cc[field_name].size, 0)

    def test_get_number_of_particles(self):
        """
        Tests the get_number_of_particles of particles.
        """
        pc = ParticleContainer(4)
        self.assertEqual(pc.get_number_of_particles(), 4)

    def test_remove_items(self):
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

        # remove items with indicies 0 and 1
        remove_arr = np.array([0, 1], dtype=np.int)
        pc.remove_items(remove_arr)

        self.assertEqual(pc.get_number_of_particles(), 2)
        self.assertEqual(check_array(pc['position-x'], [1.0, 4.0]), True)
        self.assertEqual(check_array(pc['position-y'], [3.0, 1.0]), True)
        self.assertEqual(check_array(pc['mass'], [3.0, 4.0]), True)
        self.assertEqual(check_array(pc['momentum-x'], [2.0, 3.0]), True)
        self.assertEqual(check_array(pc['momentum-y'], [1.0, 1.0]), True)
        self.assertEqual(check_array(pc['energy'], [1.0, 1.0]), True)

        # now try invalid operations to make sure errors are raised
        remove = np.arange(10, dtype=np.int)
        self.assertRaises(ValueError, pc.remove_items, remove)

        remove = np.array([2], dtype=np.int)
        pc.remove_items(remove)

        # make sure no change has occured
        self.assertEqual(pc.get_number_of_particles(), 2)
        self.assertEqual(check_array(pc['position-x'], [1.0, 4.0]), True)
        self.assertEqual(check_array(pc['position-y'], [3.0, 1.0]), True)
        self.assertEqual(check_array(pc['mass'], [3.0, 4.0]), True)
        self.assertEqual(check_array(pc['momentum-x'], [2.0, 3.0]), True)
        self.assertEqual(check_array(pc['momentum-y'], [1.0, 1.0]), True)
        self.assertEqual(check_array(pc['energy'], [1.0, 1.0]), True)

    def test_remove_tagged_particles(self):
        """
        Tests the remove_tagged_particles function.
        """

        pc = ParticleContainer(4)
        pc['position-x'][:] = [1., 2., 3., 4.]
        pc['position-y'][:] = [0., 1., 2., 3.]
        pc['mass'][:] = [1., 1., 1., 1.]
        pc['tag'][:] = [1, 0, 1, 1]

        pc.remove_tagged_particles(0)

        self.assertEqual(pc.get_number_of_particles(), 3)
        self.assertEqual(check_array(pc['position-x'], [1., 4., 3.]), True)
        self.assertEqual(check_array(pc['position-y'], [0., 3., 2.]), True)
        self.assertEqual(check_array(pc['mass'], [1., 1., 1.]), True)
        self.assertEqual(check_array(pc['tag'], [1, 1, 1]), True)

    def test_align_particles(self):
        """
        Tests the align particles function.
        """
        pc = ParticleContainer(10)
        pc['position-x'][:] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pc['position-y'][:] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        pc['tag'][:] = [0, 0, 1, 1, 1, 0, 4, 0, 1, 5]

        pc.align_particles()
        self.assertEqual(check_array(pc['position-x'],
                                     [1, 2, 6, 8, 5, 3, 7, 4, 9, 10]), True)
        self.assertEqual(check_array(pc['position-y'],
                                     [10, 9, 5, 3, 6, 8, 4, 7, 2, 1]), True)

        # should do nothing
        pc['tag'][:] = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        pc.align_particles()
        self.assertEqual(check_array(pc['position-x'],
                                     [1, 2, 6, 8, 5, 3, 7, 4, 9, 10]), True)
        self.assertEqual(check_array(pc['position-y'],
                                     [10, 9, 5, 3, 6, 8, 4, 7, 2, 1]), True)

    def test_extend(self):
        """
        Tests the extend function.
        """
        cc = CarrayContainer(var_dict={"tmp": "int"})
        self.assertEqual(cc.get_number_of_items(), 0)
        print "num:", cc.get_number_of_items()

        cc.extend(100)

        print "num:", cc.get_number_of_items()
        self.assertEqual(cc.get_number_of_items(), 100)
        for field in cc.properties.itervalues():
            self.assertEqual(field.length, 100)

    def test_extract_particles(self):
        """
        Tests the extract particles function.
        """
        pc = ParticleContainer(10)
        pc['position-x'][:] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pc['position-y'][:] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        pc['tag'][:] = [0, 0, 1, 1, 1, 0, 4, 0, 1, 5]

        indices = np.array([5, 1, 7, 3, 9])
        pc2 = pc.extract_items(indices)

        self.assertEqual(check_array(pc2['position-x'],
                                     [6, 2, 8, 4, 10]), True)
        self.assertEqual(check_array(pc2['position-y'],
                                     [5, 9, 3, 7, 1]), True)
        self.assertEqual(check_array(pc2['tag'],
                                     [0, 0, 0, 1, 5]), True)

    def test_append_parray(self):
        """
        Tests the append parray function.
        """
        pc1 = ParticleContainer(5)
        pc1['position-x'][:] = [1, 2, 3, 4, 5]
        pc1['position-y'][:] = [10, 9, 8, 7, 6]
        pc1['tag'][:] = [0, 0, 0, 0, 0]

        pc2 = ParticleContainer(5)
        pc2['position-x'][:] = [6, 7, 8, 9, 10]
        pc2['position-y'][:] = [5, 4, 3, 2, 1]
        pc2['tag'][:] = [1, 1, 1, 1, 1]

        pc1.append_container(pc2)

        self.assertEqual(check_array(pc1['position-x'],
                                     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), True)
        self.assertEqual(check_array(pc1['position-y'],
                                     [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]), True)
        self.assertEqual(check_array(pc1['tag'],
                                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), True)
