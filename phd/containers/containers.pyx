import numpy as np
cimport numpy as np

from ..utils.particle_tags import ParticleTAGS
from ..utils.carray cimport BaseArray, DoubleArray, IntArray, LongArray, LongLongArray
from cpython cimport PyDict_Contains, PyDict_GetItem


cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost

cdef class CarrayContainer:

    # add a method to add extra fields
    def __init__(self, int num_items=0, dict var_dict=None):
        """
        Create container of carrays of size num_items

        Parameters
        ----------
        num_items : int
            number of items that the carray's will hold
        """
        cdef str name, dtype

        self.properties = {}
        self.carray_info = {}

        if var_dict != None:
            for name in var_dict:
                dtype = var_dict[name]
                self.register_property(num_items, name, dtype)

    def register_property(self, int size, str name, str dtype="double"):
        """
        Register new carray

        Parameters
        ----------
        size : int
            size of the carray
        name : str
            name of carray to be added
        dtype : str
            data type of carray
        """
        if name in self.properties.keys():
            raise RuntimeError("Carray already registered")

        if len(self.properties) != 0:
            if size != self.get_number_of_items():
                raise RuntimeError("size inconsistent with carray size")

        # store data type of field
        self.carray_info[name] = dtype

        if dtype == "double":
            self.properties[name] = DoubleArray(size)
        elif dtype == "int":
            self.properties[name] = IntArray(size)
        elif dtype == "long":
            self.properties[name] = LongArray(size)
        elif dtype == "longlong":
            self.properties[name] = LongLongArray(size)
        else:
            raise ValueError("Unrecognized dtype: %s" % dtype)

    def __getitem__(self, str name):
        """
        Access carrays as numpy array

        Parameters
        ----------
        name : str
            name of carray retreive
        """
        keys = self.properties.keys()
        if name in keys:
            return self.properties[name].get_npy_array()
        else:
            raise AttributeError("Unrecognized field: %s" % name)

    cpdef int get_number_of_items(self):
        """Return the number of items in carray"""
        if len(self.properties) > 0:
            return self.properties.values()[0].length
        else:
            return 0

    cpdef extend(self, int num_items):
        """
        Increase the total number of items by requested amount.
        """
        if num_items <= 0:
            return

        cdef int old_size = self.get_number_of_items()
        cdef int new_size = old_size + num_items
        cdef BaseArray arr

        for arr in self.properties.values():
            arr.resize(new_size)

    cpdef BaseArray get_carray(self, str prop):
        """Return the c-array for the property."""
        if PyDict_Contains(self.properties, prop) == 1:
            return <BaseArray> PyDict_GetItem(self.properties, prop)
        else:
            return None

    cdef _check_property(self, str prop):
        """Check if a property is present or not."""
        if PyDict_Contains(self.properties, prop):
                return
        else:
            raise AttributeError, 'property %s not present' % (prop)

    cpdef resize(self, int size):
        """Resize all arrays to the new size."""
        cdef BaseArray array
        for array in self.properties.values():
            array.resize(size)

    def get_sendbufs(self, np.ndarray indices):
        cdef str prop
        cdef dict sendbufs = {}

        for prop in self.properties:
            sendbufs[prop] = self[prop][indices]

        return sendbufs

    cpdef int append_container(self, CarrayContainer container):
        """
        Add items from a carray container.

        Properties that are not there in self will be added.
        """
        if container.get_number_of_items() == 0:
            return 0

        cdef int num_extra_items = container.get_number_of_items()
        cdef int old_num_items = self.get_number_of_items()
        cdef str prop_name
        cdef BaseArray dest, source
        cdef np.ndarray nparr_dest, nparr_source

        # extend current arrays by the required number of items
        self.extend(num_extra_items)

        # should check that fields are equal or not error
        for prop_name in container.properties.keys():
            if PyDict_Contains(self.properties, prop_name):
                dest = <BaseArray> PyDict_GetItem(self.properties, prop_name)
                source = <BaseArray> PyDict_GetItem(container.properties, prop_name)
                nparr_source = source.get_npy_array()
                nparr_dest = dest.get_npy_array()
                nparr_dest[old_num_items:] = nparr_source

    cpdef CarrayContainer extract_items(self, np.ndarray index_array, list fields=None):
        """
        Create new carray container for item indices in index_array

        Parameters
        ----------

        index_array : np.ndarray
            Indices of items to be extracted.
        props : list
            The list of properties to extract, if None all properties
            are extracted.
        """
        cdef CarrayContainer result_array = CarrayContainer()
        cdef BaseArray dst_prop_array, src_prop_array
        cdef list prop_names
        cdef str prop_type, prop, dtype
        cdef int size = index_array.size

        if fields is None:
            prop_names = self.properties.keys()
        else:
            prop_names = fields

        # now we have the result array setup
        # resize it
        if size == 0:
            return result_array

        # allocate carrays
        for prop in prop_names:
            dtype = self.carray_info[prop]
            result_array.register_property(size, prop, dtype)

        # copy the required indices for each property
        for prop in prop_names:
            src_prop_array = self.get_carray(prop)
            dst_prop_array = result_array.get_carray(prop)
            src_prop_array.copy_values(index_array, dst_prop_array)

        return result_array

    cpdef remove_items(self, np.ndarray index_list):
        """
        Remove items whose indices are given in index_list.

        We repeatedly interchange the values of the last element and values from
        the index_list and reduce the size of the array by one. This is done for
        every property and temporary arrays that is being maintained.

        Parameters
        ---------
        index_list : np.ndarray
            array of indices, this array should be a LongArray
        """
        cdef str msg
        cdef np.ndarray sorted_indices
        cdef BaseArray prop_array
        cdef int num_arrays, i
        cdef list temp_arrays
        cdef list property_arrays

        if index_list.size > self.get_number_of_items():
            msg = 'Number of items to be removed is greater than'
            msg += 'number of items in array'
            raise ValueError, msg

        sorted_indices = np.sort(index_list)
        num_arrays = len(self.properties.keys())

        property_arrays = self.properties.values()

        for i in range(num_arrays):
            prop_array = property_arrays[i]
            prop_array.remove(sorted_indices, 1)

    cdef void extract_field_vec_ptr(self, np.float64_t *vec[3], str field_name):

        cdef int i
        cdef str axis, field
        cdef DoubleArray arr

        for i, axis in enumerate("xyz"):
            field = field_name + "-" + axis
            if field in self.properties.keys():
                arr = <DoubleArray> self.get_carray(field)
                vec[i] = arr.get_data_ptr()


cdef class ParticleContainer(CarrayContainer):

    # add a method to add extra fields
    def __init__(self, int num_real_parts=0, int dim=2, dict var_dict=None):
        """
        Create a particle array with property arrays of size
        num_real_particles

        Parameters
        ----------
        num_real_particles : int
            number of real particles that the particle array will hold
        """
        cdef str name, dtype

        self.num_real_particles = num_real_parts
        self.num_ghost_particles = 0
        self.properties = {}
        self.dim = dim

        self.carray_info = {}

        if var_dict == None:

            # register position
            self.register_property(num_real_parts, "position-x", "double")
            self.register_property(num_real_parts, "position-y", "double")

            # register primitive fields
            self.register_property(num_real_parts, "density", "double")
            self.register_property(num_real_parts, "velocity-x", "double")
            self.register_property(num_real_parts, "velocity-y", "double")
            self.register_property(num_real_parts, "pressure", "double")

            # register conservative fields
            self.register_property(num_real_parts, "mass", "double")
            self.register_property(num_real_parts, "momentum-x", "double")
            self.register_property(num_real_parts, "momentum-y", "double")
            self.register_property(num_real_parts, "energy", "double")

            # information for prallel runs
            self.register_property(num_real_parts, "key", "longlong")
            self.register_property(num_real_parts, "process", "long")

            # particle labels 
            self.register_property(num_real_parts, "tag", "int")
            self.register_property(num_real_parts, "type", "int")
            self.register_property(num_real_parts, "ids", "long")
            self.register_property(num_real_parts, "map", "long")

            # particle geometry
            self.register_property(num_real_parts, "w-x", "double")
            self.register_property(num_real_parts, "w-y", "double")
            self.register_property(num_real_parts, "dcom-x", "double")
            self.register_property(num_real_parts, "dcom-y", "double")
            self.register_property(num_real_parts, "volume", "double")
            self.register_property(num_real_parts, "radius", "double")

            if dim == 3:
                self.register_property(num_real_parts, "position-z", "double")
                self.register_property(num_real_parts, "velocity-z", "double")
                self.register_property(num_real_parts, "momentum-z", "double")
                self.register_property(num_real_parts, "w-z", "double")
                self.register_property(num_real_parts, "dcom-z", "double")

            # set initial particle tags to be real
            self['tag'][:] = Real

        else:

            for name in var_dict:
                dtype = var_dict[name]
                self.register_property(num_real_parts, name, dtype)


    cpdef int get_number_of_particles(self, bint real=False):
        """Return the number of particles"""
        if real:
            return self.num_real_particles
        else:
            if len(self.properties) > 0:
                return self.properties.values()[0].length
            else:
                return 0

    cpdef remove_tagged_particles(self, np.int8_t tag):
        """Remove particles that have the given tag.

        Parameters
        ----------

        tag : int8
            The type of particles that need to be removed.
        """
        cdef LongArray indices = LongArray()
        cdef IntArray tag_array = self.properties['tag']
        cdef np.int8_t* tagarrptr = tag_array.get_data_ptr()
        cdef np.ndarray ind
        cdef int i

        # find the indices of the particles to be removed
        for i in range(tag_array.length):
            if tagarrptr[i] == tag:
                indices.append(i)

        # remove the particles
        ind = indices.get_npy_array()
        self.remove_items(ind)


    cpdef int align_particles(self) except -1:
        """Moves all Real particles to the beginning of the array.

        This makes retrieving numpy slices of properties of Real
        particles possible. This facility will be required frequently.
        """
        cdef size_t i, num_particles
        cdef size_t next_insert
        cdef size_t num_arrays
        cdef int tmp
        cdef LongArray index_array
        cdef IntArray tag_arr
        cdef BaseArray arr
        cdef list arrays
        cdef int num_real_particles = 0
        cdef int num_moves = 0

        next_insert = 0
        num_particles = self.get_number_of_particles()

        tag_arr = self.get_carray('tag')

        # malloc the new index array
        index_array = LongArray(num_particles)

        for i in range(num_particles):
            if tag_arr.data[i] == Real:
                num_real_particles += 1
                if i != next_insert:
                    tmp = index_array.data[next_insert]
                    index_array.data[next_insert] = i
                    index_array.data[i] = tmp
                    next_insert += 1
                    num_moves += 1
                else:
                    index_array.data[i] = i
                    next_insert += 1
            else:
                index_array.data[i] = i

        self.num_real_particles = num_real_particles
        self.num_ghost_particles = num_particles - num_real_particles

        # we now have the alinged indices. Rearrange the particles
        # accordingly
        arrays = self.properties.values()
        num_arrays = len(arrays)

        for i in range(num_arrays):
            arr = arrays[i]
            arr.align_array(index_array.get_npy_array())
