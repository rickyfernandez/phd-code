import unittest
import numpy as np
from load_balance.tree import QuadTree
from hilbert.hilbert import py_hilbert_key_2d


class TestQuadTreeFirstLevel(unittest.TestCase):

    def setUp(self):

        order = 1
        self.particles = np.array([[0.25, 0.25, 0.75, 0.75],
                                   [0.25, 0.75, 0.75, 0.25]], dtype=np.float64)

        self.num_particles = self.particles.shape[1]

        keys = np.array([py_hilbert_key_2d(p, order)
            for p in (self.particles.T*2**order).astype(dtype=np.int32)])

        self.sorted_keys = np.ascontiguousarray(keys[keys.argsort()]).astype(np.int64)
        corner = np.zeros(2)

        self.tree = QuadTree(self.num_particles, self.sorted_keys,
                corner, 1.0, order=order)

        self.tree.build_tree(1)

    def test_number_of_leaves(self):

        self.assertEqual(self.tree.count_leaves(), 4)

    def test_number_of_nodes(self):

        self.assertEqual(self.tree.count_leaves() + 1, 5)
        self.assertEqual(self.tree.count_nodes(), 5)

    def test_collect_leaves(self):

        keys, num_part = self.tree.collect_leaves_for_export()
        self.assertEqual(np.sum(num_part), self.num_particles)

class TestQuadTreeSecondLevel(unittest.TestCase):

    def setUp(self):

        order = 2
        self.particles = np.array([[0.25, 0.125, 0.375, 0.75, 0.75],
                                   [0.25, 0.875, 0.875, 0.75, 0.25]], dtype=np.float64)

        self.num_particles = self.particles.shape[1]

        keys = np.array([py_hilbert_key_2d(p, order)
            for p in (self.particles.T*2**order).astype(dtype=np.int32)])

        self.sorted_keys = np.ascontiguousarray(keys[keys.argsort()]).astype(np.int64)

        corner = np.zeros(2)
        self.tree = QuadTree(self.num_particles, self.sorted_keys,
                corner, 1.0, order=order)

        self.tree.build_tree(1)

    def test_number_of_leaves(self):

        self.assertEqual(self.tree.count_leaves(), 7)

    def test_number_of_nodes(self):

        self.assertEqual(self.tree.count_leaves() + 2, 9)
        self.assertEqual(self.tree.count_nodes(), 9)

    def test_collect_leaves(self):

        keys, num_part = self.tree.collect_leaves_for_export()
        self.assertEqual(np.sum(num_part), self.num_particles)


class TestQuadTreeCreateTreeTwice(unittest.TestCase):

    def setUp(self):

        order = 2
        self.particles1 = np.array([[0.125, 0.125, 0.625, 0.625],
                                    [0.25,  0.75,  0.75,  0.25]], dtype=np.float64)
        self.particles2 = np.array([[0.375, 0.375, 0.875, 0.875],
                                    [0.25,  0.75,  0.75,  0.25]], dtype=np.float64)

        self.num_particles = self.particles1.shape[1] + self.particles2.shape[1]

        keys1 = np.array([py_hilbert_key_2d(p, order)
            for p in (self.particles1.T*2**order).astype(dtype=np.int32)])

        keys2 = np.array([py_hilbert_key_2d(p, order)
            for p in (self.particles2.T*2**order).astype(dtype=np.int32)])

        self.sorted_keys1 = np.ascontiguousarray(keys1[keys1.argsort()]).astype(np.int64)
        self.sorted_keys2 = np.ascontiguousarray(keys2[keys2.argsort()]).astype(np.int64)

        corner = np.zeros(2)

        self.tree1 = QuadTree(self.num_particles, self.sorted_keys1,
                corner, 1.0, total_num_process=2, factor=1.0, order=order)
        self.tree2 = QuadTree(self.num_particles, self.sorted_keys2,
                corner, 1.0, total_num_process=2, factor=1.0, order=order)

        self.tree1.build_tree()
        self.tree2.build_tree()

        keys1, num_part1 = self.tree1.collect_leaves_for_export()
        keys2, num_part2 = self.tree2.collect_leaves_for_export()

        self.leaf_keys = np.concatenate([keys1, keys2])
        self.leaf_num_part = np.concatenate([num_part1, num_part2])

        ind = self.leaf_keys.argsort()
        self.sorted_leaf_keys = self.leaf_keys[ind]
        self.leaf_num_part = self.leaf_num_part[ind]

        self.tree3 = QuadTree(self.num_particles, self.sorted_keys1,
                corner, 1.0,
                self.sorted_leaf_keys, self.leaf_num_part,
                total_num_process=2, factor=1.0, order=order)
        self.tree3.build_tree()

    def test_number_of_leaves(self):

        self.assertEqual(self.tree3.count_leaves(), 4)

    def test_number_of_nodes(self):

        self.assertEqual(self.tree3.count_leaves() + 1, 5)
        self.assertEqual(self.tree3.count_nodes(), 5)

    def count_total_global_particles(self):

        num_leaves = self.tree3.assign_leaves_to_array()
        work = np.empty(num_leaves, dtype=np.int32)
        self.tree3.calculate_work(work)

        self.assertEqual(np.sum(work), self.num_particles)

if __name__ == "__main__":
    unittest.main()
