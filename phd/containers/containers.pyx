import numpy as np
cimport numpy as np

from cpython cimport PyDict_Contains, PyDict_GetItem

from ..utils.particle_tags import ParticleTAGS
from ..utils.carray cimport BaseArray, DoubleArray, IntArray, LongArray, LongLongArray


cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost

cdef class CarrayContainer:

    def __init__(self, int num_items=0, dict var_dict=None, **kwargs):
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
        self.named_groups = {}

        if var_dict != None:
            for name in var_dict:
                dtype = var_dict[name]
                self.register_property(num_items, name, dtype)

    cpdef register_property(self, int size, str name, str dtype="double"):
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
            raise KeyError, 'c-array %s not present' % (prop)

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

    cpdef CarrayContainer extract_items(self, LongArray index_array, list fields=None):
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
        cdef long size = index_array.length

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

    cpdef copy(self, CarrayContainer container, LongArray indices, list properties):
        """
        Copy values at indices from container. Self will be resized all contents
        will be overwritten.
        """
        cdef str prop
        cdef BaseArray dst_prop_array, src_prop_array

        # resize array
        self.resize(indices.length)

        # copy the required indices for each property
        for prop in properties:
            dst_prop_array = self.get_carray(prop)
            src_prop_array = container.get_carray(prop)
            src_prop_array.copy_values(indices, dst_prop_array)

    cpdef paste(self, CarrayContainer container, LongArray indices, list properties):
        cdef str prop
        cdef BaseArray dst_prop_array, src_prop_array

        # resize array
        #self.resize(indices.length)

        # copy the required indices for each property
        for prop in properties:
            dst_prop_array = self.get_carray(prop)
            src_prop_array = container.get_carray(prop)
            src_prop_array.paste_values(indices, dst_prop_array)

    cpdef add(self, CarrayContainer container, LongArray indices, list properties):
        cdef str prop
        cdef BaseArray dst_prop_array, src_prop_array

        # resize array
        #self.resize(indices.length)

        # copy the required indices for each property
        for prop in properties:
            dst_prop_array = self.get_carray(prop)
            src_prop_array = container.get_carray(prop)
            src_prop_array.add_values(indices, dst_prop_array)

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

    cdef void pointer_groups(self, np.float64_t *vec[], list field_names):
        cdef int i
        cdef str field, msg
        cdef DoubleArray arr

        i = 0
        for field in field_names:
            if field in self.properties.keys():
                arr = <DoubleArray> self.get_carray(field)
                vec[i] = arr.get_data_ptr()
                i += 1
            else:
                msg = 'Unknown field in pointer_groups'
                raise ValueError, msg








#        if "species" in particles.named_groups.keys():
#            named_groups["colors"] = named_groups["species"]
#            if "passive-scalars" in particles.named_groups.keys():
#            named_groups["colors"] += named_groups["passive-scalars"]
#
#        elif "passive-scalars" in particles.named_groups.keys():
#            self.do_colors = True
#            named_groups["colors"] = named_groups["species"]
