import numpy as np
from unittest import TestCase, main

from phd.containers.containers import CarrayContainer
from phd.utils.carray import LongArray

def check_array(x, y):
    """Check if two array are equal with an absolute tolerance of
    1e-16."""
    return np.allclose(x, y , atol=1e-16, rtol=0)


class TestCarrayContainer(TestCase):
    """Tests for the ParticleArray class."""
    def setUp(self):
        self.carrays = {
                "mass": "double",
                "momentum-x": "double",
                "momentum-y": "double",
                "energy": "double",
                }

    def test_constructor_dict(self):
        """Test the constructor using dict."""
        container = CarrayContainer(carrays_to_register=self.carrays)

        self.assertEqual(container.get_carray_size(), 0)
        self.assertItemsEqual(container.carrays.keys(), self.carrays.keys())

        for carray_name in container.carrays.keys():
            self.assertEqual(container[carray_name].size, 0)

    def test_register_carray(self):
        """Test carray registration."""
        container = CarrayContainer()

        container.register_carray(5, "x", "double")
        self.assertEqual(container.get_carray_size(), 5)
        self.assertTrue("x" in container.carrays.keys())
        self.assertEqual(len(container.carrays), 1)
        self.assertEqual(len(container.carray_dtypes), 1)
        self.assertEqual(container.carray_dtypes["x"], "double")

        container.register_carray(5, "y", "int")
        self.assertEqual(container.get_carray_size(), 5)
        self.assertTrue("y" in container.carrays.keys())
        self.assertEqual(len(container.carrays), 2)
        self.assertEqual(len(container.carray_dtypes), 2)
        self.assertEqual(container.carray_dtypes["y"], "int")

        container.register_carray(5, "z", "long")
        self.assertEqual(container.get_carray_size(), 5)
        self.assertTrue("z" in container.carrays.keys())
        self.assertEqual(len(container.carrays), 3)
        self.assertEqual(len(container.carray_dtypes), 3)
        self.assertEqual(container.carray_dtypes["z"], "long")

        container.register_carray(5, "w", "longlong")
        self.assertEqual(container.get_carray_size(), 5)
        self.assertTrue("w" in container.carrays.keys())
        self.assertEqual(len(container.carrays), 4)
        self.assertEqual(len(container.carray_dtypes), 4)
        self.assertEqual(container.carray_dtypes["w"], "longlong")

        self.assertRaises(RuntimeError, container.register_carray, 5, "w", "longlong")
        self.assertRaises(ValueError, container.register_carray, 5, "u", "badtype")

    def test_get_carray_size(self):
        """Tests the get_carray_size of particles."""
        carrays = {
                "mass": "double",
                "momentum-x": "double",
                "momentum-y": "double",
                "energy": "double",
                }

        container = CarrayContainer(4, self.carrays)
        self.assertEqual(container.get_carray_size(), 4)

    def test_remove_items(self):
        """"Test remove selected items."""
        container = CarrayContainer(4, self.carrays)

        container['mass'][:]       = [1.0, 2.0, 3.0, 4.0]
        container['momentum-x'][:] = [0.0, 1.0, 2.0, 3.0]
        container['momentum-y'][:] = [1.0, 1.0, 1.0, 1.0]
        container['energy'][:]     = [1.0, 1.0, 1.0, 1.0]

        # remove items with indicies 0 and 1
        remove_arr = np.array([0, 1], dtype=np.int)
        container.remove_items(remove_arr)

        self.assertEqual(container.get_carray_size(), 2)
        self.assertEqual(check_array(container['mass']      , [3.0, 4.0]), True)
        self.assertEqual(check_array(container['momentum-x'], [2.0, 3.0]), True)
        self.assertEqual(check_array(container['momentum-y'], [1.0, 1.0]), True)
        self.assertEqual(check_array(container['energy']    , [1.0, 1.0]), True)

        # now try invalid operations to make sure errors are raised
        remove = np.arange(10, dtype=np.int)
        self.assertRaises(ValueError, container.remove_items, remove)

        remove = np.array([2], dtype=np.int)
        container.remove_items(remove)

        # make sure no change has occured
        self.assertEqual(container.get_carray_size(), 2)
        self.assertEqual(check_array(container['mass']      , [3.0, 4.0]), True)
        self.assertEqual(check_array(container['momentum-x'], [2.0, 3.0]), True)
        self.assertEqual(check_array(container['momentum-y'], [1.0, 1.0]), True)
        self.assertEqual(check_array(container['energy']    , [1.0, 1.0]), True)

    def test_remove_tagged_particles(self):
        """Tests the remove_tagged_particles function."""

        container = CarrayContainer(4, {"x": "double", "y": "double", "z": "double", "tag": "int"})
        container["x"][:] = [1., 2., 3., 4.]
        container["y"][:] = [0., 1., 2., 3.]
        container["z"][:] = [1., 1., 1., 1.]
        container["tag"][:] = [1, 0, 1, 1]

        container.remove_tagged_particles(0)

        self.assertEqual(container.get_carray_size(), 3)
        self.assertEqual(check_array(container["x"], [1., 4., 3.]), True)
        self.assertEqual(check_array(container["y"], [0., 3., 2.]), True)
        self.assertEqual(check_array(container["z"], [1., 1., 1.]), True)
        self.assertEqual(check_array(container["tag"], [1, 1, 1]), True)

    def test_resize(self):
        """Tests the resize function."""
        container = CarrayContainer(20, carrays_to_register={"tmp": "int"})
        self.assertEqual(container.get_carray_size(), 20)

        container.resize(42)

        self.assertEqual(container.get_carray_size(), 42)
        for field in container.carrays.itervalues():
            self.assertEqual(field.length, 42)

        self.assertRaises(RuntimeError, container.resize, -1)

        container.resize(0)
        self.assertEqual(container.get_carray_size(), 0)

    def test_extend(self):
        """Tests the extend function."""
        container = CarrayContainer(carrays_to_register={"tmp": "int"})
        self.assertEqual(container.get_carray_size(), 0)

        container.extend(100)

        self.assertEqual(container.get_carray_size(), 100)
        for field in container.carrays.itervalues():
            self.assertEqual(field.length, 100)

        self.assertRaises(RuntimeError, container.extend, -1)

        container.extend(0)
        self.assertEqual(container.get_carray_size(), 100)

    def test_extract_items(self):
        """Tests the extract items from carrays."""
        container = CarrayContainer(10, {"x": "double", "y": "double", "z": "double"})
        container["x"][:] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        container["y"][:] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        container["z"][:] = [0, 0, 1, 1, 1, 0, 4, 0, 1, 5]

        indices = LongArray(5)
        ind_npy = indices.get_npy_array()
        ind_npy[:] = np.array([5, 1, 7, 3, 9])

        container2 = container.extract_items(indices)

        self.assertEqual(check_array(
            container2["x"], [6, 2, 8, 4, 10]), True)

        self.assertEqual(check_array(
            container2["y"], [5, 9, 3, 7, 1]), True)

        self.assertEqual(check_array(
            container2["z"], [0, 0, 0, 1, 5]), True)

        # remove selected carrays
        container2 = container.extract_items(indices, ["x", "z"])

        self.assertEqual(check_array(
            container2["x"], [6, 2, 8, 4, 10]), True)

        self.assertEqual(check_array(
            container2["z"], [0, 0, 0, 1, 5]), True)

    def test_append_container(self):
        """Tests the append container function."""
        container = CarrayContainer(5, {"x": "int", "y": "int"})
        container["x"][:] = [1, 2, 3, 4, 5]
        container["y"][:] = [10, 9, 8, 7, 6]

        container2 = CarrayContainer(5, {"x": "int", "y": "int"})
        container2["x"][:] = [6, 7, 8, 9, 10]
        container2["y"][:] = [5, 4, 3, 2, 1]

        container.append_container(container2)

        self.assertEqual(check_array(
            container["x"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), True)
        self.assertEqual(check_array(
            container["y"], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]), True)

    def test_copy(self):
        """Tests copy function."""
        container = CarrayContainer(5, {"x": "int", "y": "int"})
        container["x"][:] = [1, 2, 3, 4, 5]
        container["y"][:] = [10, 9, 8, 7, 6]

        container2 = CarrayContainer(5, {"x": "int", "y": "int"})
        container2["x"][:] = [6, 7, 8, 9, 10]
        container2["y"][:] = [5, 4, 3, 2, 1]

        indices = LongArray(3)
        ind_npy = indices.get_npy_array()
        ind_npy[:] = np.array([1, 4, 3])

        container.copy(container2, indices, ["x", "y"])
        self.assertEqual(container.get_carray_size(), 3)

        self.assertEqual(check_array(container["x"], [7, 10, 9]), True)
        self.assertEqual(check_array(container["y"], [4, 1, 2]), True)

    def test_paste(self):
        """Tests paste function."""
        container = CarrayContainer(5, {"x": "int", "y": "int"})
        container["x"][:] = [1, 2, 3, 4, 5]
        container["y"][:] = [10, 9, 8, 7, 6]

        container2 = CarrayContainer(3, {"x": "int", "y": "int"})
        container2["x"][:] = [6, 7, 8]
        container2["y"][:] = [5, 4, 3]

        indices = LongArray(3)
        ind_npy = indices.get_npy_array()
        ind_npy[:] = np.array([1, 4, 3])

        container.paste(container2, indices, ["x", "y"])
        self.assertEqual(container.get_carray_size(), 5)

        self.assertEqual(check_array(container["x"], [1, 6, 3, 8, 7]), True)
        self.assertEqual(check_array(container["y"], [10, 5, 8, 3, 4]), True)

    def test_add(self):
        """Tests add function."""
        container = CarrayContainer(5, {"x": "int", "y": "int"})
        container["x"][:] = [1, 2, 3, 4, 5]
        container["y"][:] = [10, 9, 8, 7, 6]

        container2 = CarrayContainer(3, {"x": "int", "y": "int"})
        container2["x"][:] = [6, 7, 8]
        container2["y"][:] = [5, 4, 3]

        indices = LongArray(3)
        ind_npy = indices.get_npy_array()
        ind_npy[:] = np.array([1, 4, 3])

        container.add(container2, indices, ["x", "y"])
        self.assertEqual(container.get_carray_size(), 5)

        self.assertEqual(check_array(container["x"], [1, 8, 3, 12, 12]), True)
        self.assertEqual(check_array(container["y"], [10, 14, 8, 10, 10]), True)

if __name__ == "__main__":
    main()
