import unittest
import numpy as np

from utils.carray import DoubleArray, IntArray

class TestDoubleArray(unittest.TestCase):
    """Tests for the DoubleArray class."""
    def test_constructor(self):
        """Test the constructor."""
        da = DoubleArray(10)

        self.assertEqual(da.length, 10)
        self.assertEqual(da.alloc, 10)
        self.assertEqual(len(da.get_npy_array()), 10)

        da = DoubleArray()

        self.assertEqual(da.length, 0)
        self.assertEqual(da.alloc, 16)
        self.assertEqual(len(da.get_npy_array()), 0)

    def test_get_set_indexing(self):
        """Test get/set and [] operator."""
        da = DoubleArray(10)
        da.set(0, 10.0)
        da.set(9, 1.0)

        self.assertEqual(da.get(0), 10.0)
        self.assertEqual(da.get(9), 1.0)

        da[9] = 2.0
        self.assertEqual(da[9], 2.0)

    def test_append(self):
        """Test the append function."""
        da = DoubleArray(0)
        da.append(1.0)
        da.append(2.0)
        da.append(3.0)

        self.assertEqual(da.length, 3)
        self.assertEqual(da[0], 1.0)
        self.assertEqual(da[1], 2.0)
        self.assertEqual(da[2], 3.0)

    def test_resize(self):
        """Tests the resize function."""
        da = DoubleArray(0)

        da.resize(20)
        self.assertEqual(da.length, 20)
        self.assertEqual(len(da.get_npy_array()), 20)
        self.assertEqual(da.alloc >= da.length, True)

    def test_get_npy_array(self):
        """Tests the get_npy_array array."""
        da = DoubleArray(3)
        da[0] = 1.0
        da[1] = 2.0
        da[2] = 3.0

        nparray = da.get_npy_array()
        self.assertEqual(len(nparray), 3)

        for i in range(3):
            self.assertEqual(nparray[i], da[i])

    def test_squeeze(self):
        """Tests the squeeze function."""
        da = DoubleArray(5)
        da.append(4.0)

        self.assertEqual(da.alloc > da.length, True)

        da.squeeze()

        self.assertEqual(da.length, 6)
        self.assertEqual(da.alloc == da.length, True)
        self.assertEqual(len(da.get_npy_array()), 6)

    def test_reset(self):
        """Tests the reset function."""
        da = DoubleArray(5)
        da.reset()

        self.assertEqual(da.length, 0)
        self.assertEqual(da.alloc, 5)
        self.assertEqual(len(da.get_npy_array()), 0)

    def test_extend(self):
        """Tests teh extend function."""
        da1 = DoubleArray(5)

        for i in range(5):
            da1[i] = i

        da2 = DoubleArray(5)

        for i in range(5):
            da2[i] = 5 + i

        da1.extend(da2.get_npy_array())

        self.assertEqual(da1.length, 10)
        self.assertEqual(np.allclose(da1.get_npy_array(), np.arange(10, dtype=np.float64)), True)

    def test_remove(self):
        """Tests the remove function"""
        da1 = DoubleArray(10)
        da1_array = da1.get_npy_array()
        da1_array[:] = np.arange(10, dtype=np.float64)

        rem = [0, 4, 3]
        da1.remove(np.array(rem, dtype=np.int))
        self.assertEqual(da1.length, 7)
        self.assertEqual(np.allclose(
            np.array([7.0, 1.0, 2.0, 8.0, 9.0, 5.0, 6.0], dtype=np.float64),
            da1.get_npy_array()),
            True)

        da1.remove(np.array(rem, dtype=np.int))
        self.assertEqual(da1.length, 4)
        self.assertEqual(np.allclose(
            np.array([6.0, 1.0, 2.0, 5.0], dtype=np.float64),
            da1.get_npy_array()),
            True)

        rem = [0, 1, 3]
        da1.remove(np.array(rem, dtype=np.int))
        self.assertEqual(da1.length, 1)
        self.assertEqual(np.allclose(
            np.array([2.0], dtype=np.float64),
            da1.get_npy_array()),
            True)

        da1.remove(np.array([0], dtype=np.int))
        self.assertEqual(da1.length, 0)
        self.assertEqual(len(da1.get_npy_array()), 0)

    def test_aling_array(self):
        """Test the align_array function."""
        da1 = DoubleArray(10)
        da1_array = da1.get_npy_array()
        da1_array[:] = np.arange(10, dtype=np.float64)

        new_indices = np.array([1, 5, 3, 2, 4, 7, 8, 6, 9, 0], dtype=np.int)
        da1.align_array(new_indices)

        self.assertEqual(np.allclose(
            np.array([1.0, 5.0, 3.0, 2.0, 4.0, 7.0, 8.0, 6.0, 9.0, 0.0], dtype=np.float64),
            da1.get_npy_array()), True)


class TestIntArray(unittest.TestCase):
    """Tests for the DoubleArray class."""
    def test_constructor(self):
        """Test the constructor."""
        ia = IntArray(10)

        self.assertEqual(ia.length, 10)
        self.assertEqual(ia.alloc, 10)
        self.assertEqual(len(ia.get_npy_array()), 10)

        ia = IntArray()

        self.assertEqual(ia.length, 0)
        self.assertEqual(ia.alloc, 16)
        self.assertEqual(len(ia.get_npy_array()), 0)

    def test_get_set_indexing(self):
        """Test get/set and [] operator."""
        ia = IntArray(10)
        ia.set(0, 10)
        ia.set(9, 1)

        self.assertEqual(ia.get(0), 10)
        self.assertEqual(ia.get(9), 1)

        ia[9] = 2
        self.assertEqual(ia[9], 2)

    def test_append(self):
        """Test the append function."""
        ia = IntArray(0)
        ia.append(1)
        ia.append(2)
        ia.append(3)

        self.assertEqual(ia.length, 3)
        self.assertEqual(ia[0], 1)
        self.assertEqual(ia[1], 2)
        self.assertEqual(ia[2], 3)

    def test_resize(self):
        """Tests the resize function."""
        ia = IntArray(0)

        ia.resize(20)
        self.assertEqual(ia.length, 20)
        self.assertEqual(len(ia.get_npy_array()), 20)
        self.assertEqual(ia.alloc >= ia.length, True)

    def test_get_npy_array(self):
        """Tests the get_npy_array array."""
        ia = IntArray(3)
        ia[0] = 1
        ia[1] = 2
        ia[2] = 3

        nparray = ia.get_npy_array()
        self.assertEqual(len(nparray), 3)

        for i in range(3):
            self.assertEqual(nparray[i], ia[i])

    def test_squeeze(self):
        """Tests the squeeze function."""
        ia = IntArray(5)
        ia.append(4)

        self.assertEqual(ia.alloc > ia.length, True)

        ia.squeeze()

        self.assertEqual(ia.length, 6)
        self.assertEqual(ia.alloc == ia.length, True)
        self.assertEqual(len(ia.get_npy_array()), 6)

    def test_reset(self):
        """Tests the reset function."""
        ia = IntArray(5)
        ia.reset()

        self.assertEqual(ia.length, 0)
        self.assertEqual(ia.alloc, 5)
        self.assertEqual(len(ia.get_npy_array()), 0)

    def test_extend(self):
        """Tests teh extend function."""
        ia1 = IntArray(5)

        for i in range(5):
            ia1[i] = i

        ia2 = IntArray(5)

        for i in range(5):
            ia2[i] = 5 + i

        ia1.extend(ia2.get_npy_array())

        self.assertEqual(ia1.length, 10)
        self.assertEqual(np.allclose(ia1.get_npy_array(), np.arange(10, dtype=np.int)), True)

    def test_remove(self):
        """Tests the remove function"""
        ia1 = IntArray(10)
        ia1_array = ia1.get_npy_array()
        ia1_array[:] = np.arange(10, dtype=np.int)

        rem = [0, 4, 3]
        ia1.remove(np.array(rem, dtype=np.int))
        self.assertEqual(ia1.length, 7)
        self.assertEqual(np.allclose(
            np.array([7, 1, 2, 8, 9, 5, 6], dtype=np.int),
            ia1.get_npy_array()),
            True)

        ia1.remove(np.array(rem, dtype=np.int))
        self.assertEqual(ia1.length, 4)
        self.assertEqual(np.allclose(
            np.array([6.0, 1.0, 2.0, 5.0], dtype=np.int),
            ia1.get_npy_array()),
            True)

        rem = [0, 1, 3]
        ia1.remove(np.array(rem, dtype=np.int))
        self.assertEqual(ia1.length, 1)
        self.assertEqual(np.allclose(
            np.array([2.0], dtype=np.int),
            ia1.get_npy_array()),
            True)

        ia1.remove(np.array([0], dtype=np.int))
        self.assertEqual(ia1.length, 0)
        self.assertEqual(len(ia1.get_npy_array()), 0)

    def test_aling_array(self):
        """Test the align_array function."""
        ia1 = IntArray(10)
        ia1_array = ia1.get_npy_array()
        ia1_array[:] = np.arange(10, dtype=np.int)

        new_indices = np.array([1, 5, 3, 2, 4, 7, 8, 6, 9, 0], dtype=np.int)
        ia1.align_array(new_indices)

        self.assertEqual(np.allclose(
            np.array([1, 5, 3, 2, 4, 7, 8, 6, 9, 0], dtype=np.int),
            ia1.get_npy_array()), True)
