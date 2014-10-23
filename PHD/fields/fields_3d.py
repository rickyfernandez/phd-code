import numpy as np
from fields_base import FieldsBase

class Fields3D(FieldsBase):
    """
    field class that holds all physical data of the particles
    """

    def __init__(self, num_real_particles, gamma, boundary):

        self.num_real_particles = num_real_particles
        self.boundary = boundary
        self.num_fields = 0
        self.field_names = []

        self.field_data = None
        self.cons = None
        self.prim = None
        self.lock = False

        self.prim_names = ["density", "velocity-x", "velocity-y", "velocity-z", "pressure"]

        self.add_field("mass")
        self.add_field("momentum-x")
        self.add_field("momentum-y")
        self.add_field("momentum-z")
        self.add_field("energy")

        self.gamma = gamma


    def update_primitive(self, vol, particles, particles_index):

        mass = self.get_field("mass")
        momx = self.get_field("momentum-x")
        momy = self.get_field("momentum-y")
        momz = self.get_field("momentum-z")
        ener = self.get_field("energy")

        self.prim = np.empty((self.num_fields, self.num_real_particles), dtype="float64")

        dens = self.get_field("density")
        velx = self.get_field("velocity-x")
        vely = self.get_field("velocity-y")
        velz = self.get_field("velocitz-y")
        pres = self.get_field("pressure")

        dens[:] = mass/vol
        velx[:] = momx/mass
        vely[:] = momy/mass
        velz[:] = momz/mass
        pres[:] = (ener/vol - 0.5*dens*(velx**2 + vely**2 + velz**2))*(self.gamma - 1.0)

        self.prim = self.boundary.primitive_to_ghost(particles, self.prim, particles_index)
