import unittest
import numpy as np

from phd.containers.containers import CarrayContainer, ParticleContainer
from phd.reconstruction.reconstruction import PieceWiseConstant
from phd.riemann.riemann import HLLC


def check_array(x, y):
    """Check if two array are equal with an absolute tolerance of
    1e-16."""
    return np.allclose(x, y , atol=1e-16, rtol=0)


class TestHLLC(unittest.TestCase):
    """Tests for the Reconstruction class."""
    def setUp(self):
        """Test reconstruction to left and right of face."""

        # setup left/right face container 
        state_vars = {
                "density": "double",
                "velocity-x": "double",
                "velocity-y": "double",
                "pressure": "double",
                }
        self.left_state  = CarrayContainer(1, var_dict=state_vars)
        self.right_state = CarrayContainer(1, var_dict=state_vars)

        self.dens_l = self.left_state['density']
        self.velx_l = self.left_state['velocity-x']
        self.vely_l = self.left_state['velocity-y']
        self.pres_l = self.left_state['pressure']

        self.dens_r = self.right_state['density']
        self.velx_r = self.right_state['velocity-x']
        self.vely_r = self.right_state['velocity-y']
        self.pres_r = self.right_state['pressure']

        # setup face container
        state_vars = {
                "normal-x": "double",
                "normal-y": "double",
                "velocity-x": "double",
                "velocity-y": "double",
                }
        self.faces = CarrayContainer(1, var_dict=state_vars)

        # the face is perpendicular to the x-axis and has velocity zero
        self.nx = self.faces["normal-x"];   self.ny = self.faces["normal-y"]
        self.wx = self.faces["velocity-x"]; self.wy = self.faces["velocity-y"]

        # setup flux container
        flux_vars = {
                "mass": "double",
                "momentum-x": "double",
                "momentum-y": "double",
                "energy": "double",
                }
        self.fluxes = CarrayContainer(1, var_dict=flux_vars)

        # compute reconstruction to face
        recon = PieceWiseConstant()
        self.hllc = HLLC(recon)

    def test_left_state(self):

        # both particles have the same value
        self.dens_l[:] = 1.0; self.dens_r[:] = 1.0
        self.velx_l[:] = 2.0; self.velx_r[:] = 2.0
        self.vely_l[:] = 0.0; self.vely_r[:] = 0.0
        self.pres_l[:] = 0.4; self.pres_r[:] = 0.4

        # the face is perpendicular to the x-axis and has velocity zero
        self.nx[0] = 1; self.ny[0] = 0
        self.wx[0] = 0; self.wy[0] = 0

        ans = {"mass": 2.0, "momentum-x": 4.4, "momentum-y": 0.0, "energy": 6.8}
        self.hllc.solve(self.fluxes, self.left_state, self.right_state, self.faces, 0., 0., 0)

        # left and right states should be the same
        for field in self.fluxes.properties.keys():
            self.assertAlmostEqual(self.fluxes[field][0], ans[field])

    def test_right_state(self):

        # both particles have the same value
        self.dens_l[:] = 1.0; self.dens_r[:] = 1.0
        self.velx_l[:] = -2.; self.velx_r[:] = -2.
        self.vely_l[:] = 0.0; self.vely_r[:] = 0.0
        self.pres_l[:] = 0.4; self.pres_r[:] = 0.4

        # the face is perpendicular to the x-axis and has velocity zero
        self.nx[0] = 1; self.ny[0] = 0
        self.wx[0] = 0; self.wy[0] = 0

        ans = {"mass": -2., "momentum-x": 4.4, "momentum-y": 0.0, "energy": -6.8}
        self.hllc.solve(self.fluxes, self.left_state, self.right_state, self.faces, 0., 0., 0)

        # left and right states should be the same
        for field in self.fluxes.properties.keys():
            self.assertAlmostEqual(self.fluxes[field][0], ans[field])
