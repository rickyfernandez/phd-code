import numpy as np

class Fields(object):

    def __init__(self, num_real_particles, gamma, boundary):

        self.num_real_particles = num_real_particles
        self.boundary = boundary
        self.num_fields = 0
        self.field_names = []

        self.field_data = None
        self.cons = None
        self.prim = None
        self.lock = False

        self.prim_names = ["density", "velocity-x", "velocity-y", "pressure"]

        self.add_field("mass")
        self.add_field("momentum-x")
        self.add_field("momentum-y")
        self.add_field("energy")

        self.gamma = gamma


    def add_field(self, name):

        if self.lock == True:
            raise RuntimeError

        self.field_names.append(name)
        self.num_fields += 1


    def create_fields(self):

        self.lock == True
        self.field_data = np.zeros((self.num_fields, self.num_real_particles), dtype="float64")


    def get_field(self, name, only_real=True):

        if name in self.field_names:
            i = self.field_names.index(name)
            if only_real == True:
                return self.field_data[i,:self.num_real_particles]
            else:
                return self.field_data[i,:]

        if name in self.prim_names:
            i = self.prim_names.index(name)
            if only_real == True:
                return self.prim[i,:self.num_real_particles]
            else:
                return self.prim[i,:]


    def update_primitive(self, vol, particles, particles_index):

        mass = self.get_field("mass")
        momx = self.get_field("momentum-x")
        momy = self.get_field("momentum-y")
        ener = self.get_field("energy")

        self.prim = np.empty((self.num_fields, self.num_real_particles), dtype="float64")

        dens = self.get_field("density")
        velx = self.get_field("velocity-x")
        vely = self.get_field("velocity-y")
        pres = self.get_field("pressure")

        dens[:] = mass/vol
        velx[:] = momx/mass
        vely[:] = momy/mass
        pres[:] = (ener/vol - 0.5*dens*(velx**2 + vely**2))*(self.gamma - 1.0)

        self.prim = self.boundary.primitive_to_ghost(particles, self.prim, particles_index)

    def update_boundaries(self, particles, particles_index, neighbor_graph, neighbor_graph_sizes):

        new_particles = self.boundary.update(particles, particles_index, neighbor_graph, neighbor_graph_sizes)
        self.num_real_particles = particles_index["real"].size

        return new_particles
