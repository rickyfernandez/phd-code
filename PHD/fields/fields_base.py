import numpy as np

class FieldsBase(object):
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

        self.prim_names = []
        self.gamma = gamma


    def add_field(self, name):

        if self.lock == True:
            raise RuntimeError

        self.field_names.append(name)
        self.num_fields += 1

    def calculate_sound_speed(self):
        pre = get_field("pressure")
        den = get_field("density")

        return np.sqrt(self.gamm*pre/den)


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

        if name == "sound speed":
            return self.calculate_sound_speed()


    def update_primitive(self, vol, particles, particles_index):
        pass


    def update_boundaries(self, particles, particles_index, graphs):

        new_particles = self.boundary.update_boundaries(particles, particles_index, graphs["neighbors"], graphs["number of neighbors"])
        self.num_real_particles = particles_index["real"].size

        return new_particles
