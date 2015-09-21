import numpy as np
cimport numpy as np

from utils.carray import BaseArray, DoubleArray, IntArray, LongArray, LongLongArray
from particles.particle_tags import ParticleTAGS

from utils.carray cimport BaseArray, DoubleArray, IntArray, LongArray, LongLongArray
from cpython cimport PyDict_Contains, PyDict_GetItem

cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost

cdef list _base_cons_fields = ['mass', 'momentum-x', 'momentum-y', 'energy']
cdef list _base_prim_fields = ['density', 'velocity-x', 'velocity-y', 'pressure']
cdef list _base_coords = ['position-x', 'position-y']
cdef list _base_tags   = ['key', 'tags', 'process']

cdef list _base_fields = _base_cons_fields + _base_prim_fields
cdef list _base_properties = _base_coords + _base_fields + _base_tags

cdef class ParticleArray:

    # add a method to add extra fields
    def __cinit__(self, int num_real_parts=0, dict var_dict=None):
        """
        Create a particle array with property arrays of size
        num_real_particles

        Parameters
        ----------
        num_real_particles : int
            number of real particles that the particle array will hold
        """
        self.num_real_particles = num_real_parts
        self.num_ghost_particles = 0
        self.properties = {}
        self.field_names = []

        cdef str coord, field, name, dtype

        if var_dict == None:

            for coord in _base_coords:
                self.register_property(coord)

            for field in _base_fields:
                self.field_names.append(field)
                self.register_property(field)

            self.register_property("key", "longlong")
            self.register_property("tag", "int")
            self.register_property("process", "long")

            # set initial particle tags to be real
            self['tag'][:] = Real

        else:

            for name in var_dict:
                dtype = var_dict[name]
                self.register_property(name, dtype)

    def register_property(self, str name, str dtype="double"):
        """
        Register new property array for particles

        Parameters
        ----------
        name : str
            name of the field to be added
        dtype : str
            data type
        """
        if name in self.properties.keys():
            raise RuntimeError("Field already registered")

        if dtype == "double":
            self.properties[name] = DoubleArray(self.num_real_particles)
        elif dtype == "int":
            self.properties[name] = IntArray(self.num_real_particles)
        elif dtype == "long":
            self.properties[name] = LongArray(self.num_real_particles)
        elif dtype == "longlong":
            self.properties[name] = LongLongArray(self.num_real_particles)
        else:
            raise ValueError("Unrecognized dtype: %s" % dtype)

    def __getitem__(self, str name):
        """
        Access particle arrays in dictionary style

        Parameters
        ----------
        name : str
            name of the field to be added
        """
        keys = self.properties.keys()
        if name in keys:
            return self.properties[name].get_npy_array()
        else:
            raise AttributeError("Unrecognized field: %s" % name)

    cpdef int get_number_of_particles(self, bint real=False):
        """Return the number of particles"""
        if real:
            return self.num_real_particles
        else:
            if len(self.properties) > 0:
                return self.properties.values()[0].length
            else:
                return 0

    cpdef remove_particles(self, np.ndarray index_list):
        """
        Remove particles whose indices are given in index_list.

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

        if index_list.size > self.get_number_of_particles():
            msg = 'Number of particles to be removed is greater than'
            msg += 'number of particles in array'
            raise ValueError, msg

        sorted_indices = np.sort(index_list)
        num_arrays = len(self.properties.keys())

        property_arrays = self.properties.values()

        for i in range(num_arrays):
            prop_array = property_arrays[i]
            prop_array.remove(sorted_indices, 1)

        if index_list.size > 0:
            self.align_particles()
            self.is_dirty = True
            self.indices_invalid = True

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
        cdef long i

        # find the indices of the particles to be removed
        for i in xrange(tag_array.length):
            if tagarrptr[i] == tag:
                indices.append(i)

        # remove the particles
        ind = indices.get_npy_array()
        self.remove_particles(ind)

    cpdef extend(self, int num_particles):
        """
        Increase the total number of particles by the requested amount.

        New particles are added at the end of the array, you may have to
        manually call align_particles later.
        """
        if num_particles <= 0:
            return

        cdef int old_size = self.get_number_of_particles()
        cdef int new_size = old_size + num_particles
        cdef BaseArray arr

        for arr in self.properties.values():
            arr.resize(new_size)

    def get(self, *args, only_real_particles=True):
        """Return the numpy array for the property names in the
        arguments.

        Parameters
        ---------
        only_real_particles : bool
            Indicates if properties of only real particles need to be
            returned or all particles to be returned. By default only
            real particles will be returned.
        args : list
            List of property names
        """
        cdef int nargs = len(args)
        cdef list result = []
        cdef str arg
        cdef int i
        cdef BaseArray arg_array

        if nargs == 0:
            return

        if only_real_particles == True:
            for i in range(nargs):
                arg = args[i]
                self._check_property(arg)

                if arg in self.properties:
                    arg_array = self.properties[arg]
                    result.append(
                            arg_array.get_npy_array()[:self.num_real_particles])
        else:
            for i in range(nargs):
                arg = args[i]
                self._check_property(arg)

                if arg in self.properties:
                    arg_array = self.properties[arg]
                    result.append(arg_array.get_npy_array())

        if nargs == 1:
            return result[0]
        else:
            return tuple(result)

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

    cpdef int align_particles(self) except -1:
        """Moves all Real particles to the beginning of the array.

        This makes retrieving numpy slices of properties of Real
        particles possible. This facility will be required frequently.
        """
        cdef size_t i, num_particles
        cdef size_t next_insert
        cdef size_t num_arrays
        cdef long tmp
        cdef LongArray index_array
        cdef IntArray tag_arr
        cdef BaseArray arr
        cdef list arrays
        cdef long num_real_particles = 0
        cdef long num_moves = 0

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

        if num_moves > 0:
            self.is_dirty = True
            self.indices_invalid = True

    cpdef resize(self, long size):
        """Resize all arrays to the new size."""
        cdef BaseArray array
        for array in self.properties.values():
            array.resize(size)

    cdef void make_ghost(self, np.float64_t x, np.float64_t y, np.int32_t proc):

        cdef long j, new_size
        cdef BaseArray array

        # check if adding a new particle exceeds causes the array size
        # to be larger then the allocated memory
        j = self.get_number_of_particles()
        if j + 1 > self.properties["mass"].alloc:
            # reserve memory by 10% of existing memory for all arrays
            new_size = int(1.1*self.get_number_of_particles())
            for array in self.properties.values():
                array.reserve(new_size)
                array.append(0)
        else:
            # makes sure that each array has added
            # one more particle
            for array in self.properties.values():
                array.append(0)

        # j is the next avaiable slot for a new ghost particle
        self["position-x"][j] = x
        self["position-y"][j] = y
        self["process"][j] = proc
        self["tag"][j] = Ghost

        # update nubmer of ghost particle
        self.num_ghost_particles += 1

    def get_sendbufs(self, np.ndarray indices):
        cdef str prop
        cdef dict sendbufs = {}

        for prop in self.properties:
            sendbufs[prop] = self[prop][indices]

        return sendbufs

#    def get_property_index(self, prop_name):
#        """Get the index of the property in the property array."""
#        return self.properties.get(prop_name)
#
#    cdef np.ndarray _get_real_particle_prop(self, str prop_name):
#        """Get the numpy arrray of property corresponding to only real
#        particles.
#
#        No checks are performed. Only call this after making sure that
#        the property required already exists and that the real particles
#        are placed in the beginning of the arrays.
#        """
#        cdef BaseArray prop_array
#        prop_array = self.properties.get(prop_name)
#        if prop_array is not None:
#            return prop_array.get_npy_array()[:self.num_real_particles]
#
    cpdef int append_parray(self, ParticleArray parray):
        """
        Add particles from a particle array.

        Properties that are not there in self will be added.
        """
        if parray.get_number_of_particles() == 0:
            return 0

        cdef int num_extra_particles = parray.get_number_of_particles()
        cdef int old_num_particles = self.get_number_of_particles()
        cdef str prop_name
        cdef BaseArray dest, source
        cdef np.ndarray nparr_dest, nparr_source

        # extend current arrays by the required number of particles
        self.extend(num_extra_particles)

        # should check that fields are equal or not error
        for prop_name in parray.properties.keys():
            if PyDict_Contains(self.properties, prop_name):
                dest = <BaseArray> PyDict_GetItem(self.properties, prop_name)
                source = <BaseArray> PyDict_GetItem(parray.properties, prop_name)
                nparr_source = source.get_npy_array()
                nparr_dest = dest.get_npy_array()
                nparr_dest[old_num_particles:] = nparr_source

        if num_extra_particles > 0:
            self.align_particles()
            self.is_dirty = True

        return 0


#    cpdef set_dirty(self, bint value):
#        """Set the is_dirty variable to given value."""
#        self.is_dirty = value
#
#    cpdef set_indices_invalid(self, bint value):
#        """Set the indices_invalid to the given value"""
#        self.indices_invalid = value
#
#    cpdef has_array(self, str arr_name):
#        """Returns true if the array arr_name is present"""
#        return self.properties.has_key(arr_name)
#
    cpdef ParticleArray extract_particles(self, np.ndarray index_array,
            list props=None):
        """
        Create new particle array for particles with indices in index_array

        Parameters
        ----------

        index_array : np.ndarray
            Indices of particles to be extracted.
        props : list
            The list of properties to extract, if None all properties
            are extracted.
        """
        cdef ParticleArray result_array = ParticleArray()
        cdef BaseArray dst_prop_array, src_prop_array
        cdef list prop_names
        cdef str prop_type, prop

        if props is None:
            prop_names = self.properties.keys()
        else:
            prop_names = props

        # need to add a way to add fields into result in case
        # this array has non default fields

        # now we have the result array setup
        # resize it
        if index_array.size == 0:
            return result_array

        result_array.resize(index_array.size)

        # copy the required indices for each property
        for prop in prop_names:
            src_prop_array = self.get_carray(prop)
            dst_prop_array = result_array.get_carray(prop)
            src_prop_array.copy_values(index_array, dst_prop_array)

        return result_array


#    cpdef set_tag(self, str tag_value, int flag_value, LongArray indices):
#        """Set property flag_name to flag_value for particles in indices."""
#        cdef Intarray tag_array = self.get_carray('tag')
#        cdef int i
#
#        for i in range(indices.length):
#            tag_array.data[indices.data[i]] = tag_value
#
#    cpdef set_pid(self, LongArray pids):
#        """Set property flag_name to flag_value for particles in indices."""
#        cdef Intarray pid_array = self.get_carray('tag')
#        cdef long i
#
#        cdef long np = self.get_number_of_particles()
#        for i in range(np):
#            pid_array.data[i] = pids[i]
#
#    cpdef set_to_zero(self, list props):
#
#        cdef long np = self.get_number_of_particles()
#        cdef long i
#
#        cdef BaseArray prop_arr
#        cdef str prop
#
#        for prop in props:
#            prop_arr = self.get_carray(prop)
#
#            for i in range(np):
#                prop_arr.data[a] = 0.0
#
#    def update_min_max(self, props=None):
#        """Update the min, max values of all properties."""
#        if props:
#            for prop in props:
#                array = self.properties[prop]
#                array.update_min_max()
#        else:
#            for array in self.properties.values():
#                array.update_min_max()
#
#    cdef make_ghost_from_real(int i, np.float64_t[:] x, np.float64_t[:] vel):
#        cdef int j
#        cdef str field
#        cdef np.float64_t mass
#
#        # check if adding a new particle exceeds array size
#        j = self.total_particles() + 1
#        if j > self.properties["mass"].alloc:
#            # resize all arrays by 10%
#            self.resize(int(1.1*self.total_particles())):
#
#        # copy values of i particle to new ghost particle
#        for field in self.field_names:
#            nparr = self.properties[field].get_npy_array()
#            nparr[j] = nparr[i]
#
#        # new position
#        self["position-x"][j] = pos[0]
#        self["position-y"][j] = pos[1]
#
#        # new momentum
#        mass = self["mass"][i]
#        self["momentum-x"][j] = mass*vel[0]
#        self["momentum-y"][j] = mass*vel[1]
#
#        # ?
#        self["key"][j] = -1
#        self["proc"][j] = -1
#
#        # update nubmer of ghost particle
#        self.num_ghost_particles += 1
#
