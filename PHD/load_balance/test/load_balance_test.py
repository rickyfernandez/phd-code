import unittest
import numpy as np

from hilbert.hilbert import hilbert_key_2d
from load_balance.load_balance import LoadBalance
from tools.run_parallel_script import get_directory, run

from nose.plugins.attrib import attr

path = get_directory(__file__)


"""
to run only parallel tests:
    $ nosetests -a parallel=True
"""

class TestLoadBalance(unittest.TestCase):

    @attr(slow=False, parallel=True)
    def test_exchange(self):
        run(filename='./exchange.py', nprocs=4, path=path)

#    def setUp(self):
#
#        order = 1
#        self.particles = np.array([[0.25, 0.25, 0.75, 0.75],
#                                   [0.25, 0.75, 0.75, 0.25]], dtype=np.float64)
#
#        self.load_bal = LoadBalance(self.particles, order)
#
#
#    def test_hilbert_keys_in_order(self):
#        for i in range(self.load_bal.sorted_keys.size):
#            self.assertEqual(self.load_bal.sorted_keys[i], i)


if __name__ == "__main__":
    unittest.main()
