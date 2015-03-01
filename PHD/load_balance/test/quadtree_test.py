import unittest
import numpy as np
from load_balance.load_balance import QuadTree
from tree.hilbert import hilbert_key_2d


class TestQuadTreeFirstLevel(unittest.TestCase):

    def setUp(self):

        order = 1
        self.particles = np.array([[0.25, 0.25, 0.75, 0.75],
                                   [0.25, 0.75, 0.75, 0.25]], dtype=np.float64)

        self.num_particles = self.particles.shape[1]

        keys = np.array([hilbert_key_2d(p[0], p[1], order)
            for p in (self.particles.T*2**order).astype(dtype=np.int64)])

        self.sorted_keys = np.ascontiguousarray(keys[keys.argsort()]).astype(np.uint64)

        self.tree = QuadTree(self.num_particles, self.sorted_keys, order=order)

        self.tree.build_tree(1)


    def test_number_of_leaves(self):

        self.assertEqual(self.tree.count_leaves(), 4)


    def test_number_of_nodes(self):

        self.assertEqual(self.tree.count_leaves() + 1, 5)
        self.assertEqual(self.tree.number_nodes, 5)

    def test_collect_leaves(self):

        keys, num_part = self.tree.collect_leaves_for_export()
        self.assertEqual(np.sum(num_part), self.num_particles)



class TestQuadTreeSecondLevel(unittest.TestCase):

    def setUp(self):

        order = 2
        self.particles = np.array([[0.25, 0.125, 0.375, 0.75, 0.75],
                                   [0.25, 0.875, 0.875, 0.75, 0.25]], dtype=np.float64)

        self.num_particles = self.particles.shape[1]

        keys = np.array([hilbert_key_2d(p[0], p[1], order)
            for p in (self.particles.T*2**order).astype(dtype=np.int64)])

        self.sorted_keys = np.ascontiguousarray(keys[keys.argsort()]).astype(np.uint64)

        self.tree = QuadTree(self.num_particles, self.sorted_keys, order=order)

        self.tree.build_tree(1)


    def test_number_of_leaves(self):

        self.assertEqual(self.tree.count_leaves(), 7)


    def test_number_of_nodes(self):

        self.assertEqual(self.tree.count_leaves() + 2, 9)
        self.assertEqual(self.tree.number_nodes, 9)


    def test_collect_leaves(self):

        keys, num_part = self.tree.collect_leaves_for_export()
        self.assertEqual(np.sum(num_part), self.num_particles)

if __name__ == "__main__":
    unittest.main()
