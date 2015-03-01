import unittest
import numpy as np
from load_balance.load_balance import LoadBalance
from tree.hilbert import hilbert_key_2d


class TestLoadBalance(unittest.TestCase):

    def setUp(self):

        order = 1
        self.particles = np.array([[0.25, 0.25, 0.75, 0.75],
                                   [0.25, 0.75, 0.75, 0.25]], dtype=np.float64)

        self.load_bal = LoadBalance(self.particles, order)


    def test_hilbert_keys_in_order(self):
        for i in range(self.load_bal.sorted_keys.size):
            self.assertEqual(self.load_bal.sorted_keys[i], i)


if __name__ == "__main__":
    unittest.main()
