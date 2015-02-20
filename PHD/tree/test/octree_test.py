import unittest
import numpy as np
from tree.octree import Octree
from tree.hilbert import hilbert_key_2d


class TestOctreeFirstLevel(unittest.TestCase):

    def setUp(self):

        order = 1
        self.particles = np.array([[0.25, 0.25, 0.75, 0.75],
                                   [0.25, 0.75, 0.75, 0.25]], dtype=np.float64)

        keys = np.array([hilbert_key_2d(p[0], p[1], order)
            for p in (self.particles.T*2**order).astype(dtype=np.int64)])

        self.sorted_keys = np.ascontiguousarray(keys[keys.argsort()])

        self.tree = Octree(sorted_particle_keys=self.sorted_keys, max_leaf_particles=1,
                order=order, process=1, number_process=1)

        self.tree.build_tree()

    def test_creation_of_first_level(self):

        self.assertEqual(self.tree.number_nodes, 5)

    def test_number_of_leaves_in_first_level(self):

        self.assertEqual(self.tree.count_leaves(), 4)



class TestOctreeSecondLevel(unittest.TestCase):

    def setUp(self):

        order = 2
        self.particles = np.array([[0.25, 0.125, 0.375, 0.75, 0.75],
                                   [0.25, 0.875, 0.875, 0.75, 0.25]], dtype=np.float64)

        keys = np.array([hilbert_key_2d(p[0], p[1], order)
            for p in (self.particles.T*2**order).astype(dtype=np.int64)])

        self.sorted_keys = np.ascontiguousarray(keys[keys.argsort()])

        self.tree = Octree(sorted_particle_keys=self.sorted_keys, max_leaf_particles=1,
                order=order, process=1, number_process=1)

        self.tree.build_tree()


    def test_creation_of_second_level(self):

        self.assertEqual(self.tree.number_nodes, 9)

    def test_number_of_leaves_in_second_level(self):

        self.assertEqual(self.tree.count_leaves(), 7)


if __name__ == "__main__":
    unittest.main()
