import numpy as np
from utils.carray import DoubleArray, IntArray, LongLongArray

_base_fields = ['mass', 'momentum-x', 'momentum-y', 'energy']
_base_coords = ['position-x', 'position-y']
_base_tags   = ['keys', 'tags']
_base_properties = _base_coords + _base_fields + _base_tags

class ParticleContainer(object):

    def __init__(self, num_particles):
        """Create a particle container with property arrays of size
        num_particles

        Parameters
        ----------
        num_particles - number of particles that the container will hold
        """
        self.num_particles = num_particles
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
            self.properties[name] = DoubleArray(self.num_particles)
        elif dtype == "int":
            self.properties[name] = IntArray(self.num_particles)
        elif dtype == "longlong":
            self.properties[name] = LongLongArray(self.num_particles)
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

    def remove_particles(self, indices):
        """Remove particles with given indices

        We repeatedly interchange the values of the last element and values from
        the indices and reduce the size of the array by one. This is done for every
        field in the container. Note the indices are sorted to matain order in the
        exchange

        Parameters
        ----------
        indices - a numpy array of particle indices to be removed"""

        if indices.size > self.num_particles:
            raise ValueError("More indices than particles")

        sorted_indices = np.sort(indices)

        for field in self.properties.values():
            field.remove(sorted_indices, 1)

        self.num_particles = field.length

    def remove_tagged_particles(self):
        pass

    def resize(self, new_size):
        """Resizes all fields to hold new_size number of particles

        Parameters
        ----------
        new_size - new number of particles"""
        if new_size <= 0:
            return

        for prop in self.properties.values():
            prop.resize(new_size)

        self.num_particles = new_size
