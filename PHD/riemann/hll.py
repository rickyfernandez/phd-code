from riemann_base import RiemannBase
import numpy as np
import riemann

class Hll(RiemannBase):

    def solver(self, left_face, right_face, fluxes, w, gamma, num_faces):
        riemann.hll(left_face, right_face, fluxes, w, gamma, num_faces)

    def get_dt(self, fields, vol, gamma):

        # grab values that correspond to real particles
        dens = fields.get_field("density")
        velx = fields.get_field("velocity-x")
        vely = fields.get_field("velocity-x")
        pres = fields.get_field("pressure")

        # sound speed
        c = np.sqrt(gamma*pres/dens)

        # calculate approx radius of each voronoi cell
        R = np.sqrt(vol/np.pi)

        dt_x = R/(abs(velx) + c)
        dt_y = R/(abs(vely) + c)

        return min(dt_x.min(), dt_y.min())
