import unittest
import numpy as np

#from load_balance.load_balance import LoadBalance
from utils.run_parallel_script import get_directory, run

from nose.plugins.attrib import attr

path = get_directory(__file__)


"""
to run only parallel tests:
    $ nosetests -a parallel=True
"""

class TestLoadBalance(unittest.TestCase):

#    @attr(slow=False, parallel=True)
#    def test_exchange(self):
#        run(filename='./exchange_particles.py', nprocs=4, path=path)

    @attr(slow=False, parallel=True)
    def test_load_balance_exchange_2d(self):
        run(filename='./load.py', nprocs=4, path=path)

    @attr(slow=False, parallel=True)
    def test_load_balance_exchange_3d(self):
        run(filename='./load3D.py', nprocs=4, path=path)

if __name__ == "__main__":
    unittest.main()
