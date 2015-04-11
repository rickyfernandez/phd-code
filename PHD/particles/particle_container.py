import numpy as np
from utils.carray import DoubleArray, IntArray, LongLongArray

_base_fields = ['mass', 'momentum-x', 'momentum-y', 'energy']
_base_coords = ['position-x', 'position-y']
_base_tags   = ['keys', 'tags']
_base_properties = _base_coords + _base_fields + _base_tags

class ParticleContainer(object):

    def __init__(self, num_real_particles):
        """Create a particle container with property arrays of size
        num_real_particles

        Parameters
        ----------
        num_real_particles - number of real particles that the container will hold
        """
        self.num_real_particles = num_real_particles
        self.num_ghost_particles = 0
        self.properties = {}
        self.field_names = []

        for coord in _base_coords:
            self.register_property(coord, field=False)

        for field in _base_fields:
            self.register_property(field)

        # register particle tag and hilbert keys
        self.register_property("tag", "int", field=False)
        self.register_property("key", "longlong", field=False)

    def register_property(self, name, dtype="double", field=True):
        """Register new property array for particles

        Parameters
        ----------
        name - name of the field to be added
        dtype - data type (double, int)
        field - flag indicating property is a field
        """
        if name in self.properties.keys():
            raise RuntimeError("Field already registered")

        if dtype == "double":
            self.properties[name] = DoubleArray(self.num_real_particles)
        elif dtype == "int":
            self.properties[name] = IntArray(self.num_real_particles)
        elif dtype == "longlong":
            self.properties[name] = LongLongArray(self.num_real_particles)
        else:
            raise ValueError("Unrecognized dtype: %s" % dtype)

        if field:
            self.field_names.append(name)

    def __getitem__(self, name):
        return self.get(name)

    def get(self, name):
        """Get property as numpy array by name"""
        keys = self.properties.keys()
        if name in keys:
            return self.properties[name].get_npy_array()
        else:
            raise AttributeError("Unrecognized field: %s" % name)

    def discard_ghost_and_export_particles(self, indices):
        """Remove real particles with given indices

        We repeatedly interchange the values of the last element and values from
        the indices and reduce the size of the array by one. This is done for every
        field in the container. Note the indices are sorted to matain order in the
        exchange

        Parameters
        ----------
        indices - a numpy array of real particle indices to be removed"""
        self.discard_ghost_particles()

        if indices.size > self.num_real_particles:
            raise ValueError("More indices than real particles")

        sorted_indices = np.sort(indices)

        # remove properties of real exported particles 
        for field in self.properties.values():
            field.remove(sorted_indices, 1)

        self.num_real_particles = field.length

    def discard_ghost_particles(self):
        """Discard ghost particles. This is simply done by setting the index of
        the last ghost particle to the index one past the last real particles.
        Also the counter of ghost particles is sent to zero.
        """
        for prop in self.properties.values():
            prop.shrink(self.num_real_particles)
        self.num_ghost_particles = 0

    def remove_tagged_particles(self):
        pass

    def resize(self, new_size):
        """Resizes all fields to size of new_size. Note that the arrays
        are larger but the length occupied by the particles is still the same.

        Parameters
        ----------
        new_size - new number of particles"""
        for prop in self.properties.values():
            prop.resize(new_size)
