import unittest
import numpy as np

from hilbert.hilbert import hilbert_key_2d
#from load_balance.load_balance import LoadBalance
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
        run(filename='./exchange_particles.py', nprocs=4, path=path)

#    @attr(slow=False, parallel=True)
#    def test_load_balance_exchange(self):
#        run(filename='./load.py', nprocs=4, path=path)

if __name__ == "__main__":
    unittest.main()
