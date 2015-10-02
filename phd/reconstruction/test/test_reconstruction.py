import unittest
import numpy as np

from phd.containers.containers import CarrayContainer, ParticleContainer
from phd.reconstruction.reconstruction import PieceWiseConstant


def check_array(x, y):
    """Check if two array are equal with an absolute tolerance of
    1e-16."""
    return np.allclose(x, y , atol=1e-16, rtol=0)


class TestPieceWiseConstant(unittest.TestCase):
    """Tests for the Reconstruction class."""
    def test_compute(self):
        """Test reconstruction to left and right of face."""

        # setup left/right face state array 
        state_vars = {
                "density": "double",
                "velocity-x": "double",
                "velocity-y": "double",
                "pressure": "double",
                }
        left_state  = CarrayContainer(1, var_dict=state_vars)
        right_state = CarrayContainer(1, var_dict=state_vars)

        # setup face information
        state_vars = {
                "pair-i": "longlong",
                "pair-j": "longlong",
                }
        faces = CarrayContainer(1, var_dict=state_vars)
        pair_i = faces["pair-i"]; pair_j = faces["pair-j"]
        pair_i[0] = 0; pair_j[0] = 1

        # setup particles that defined the face
        pc = ParticleContainer(2)

        dens = pc['density']
        velx = pc['velocity-x']
        vely = pc['velocity-y']
        pres = pc['pressure']

        # both particles have the same value
        dens[:] = 1.0
        velx[:] = 1.5
        vely[:] = 5.0
        pres[:] = 0.1

        # compute reconstruction to face
        gamma = 1.4
        recon = PieceWiseConstant()
        recon.compute(pc, faces, left_state, right_state, gamma, 0.1)

        # left and right states should be the same
        for field in left_state.properties.keys():
            self.assertAlmostEqual(left_state[field], pc[field][0])
            self.assertAlmostEqual(left_state[field], pc[field][1])
