import numpy as np
cimport numpy as np

from cpython cimport PyDict_Contains, PyDict_GetItem

from ..utils.particle_tags import ParticleTAGS
from ..utils.carray cimport BaseArray, DoubleArray, IntArray, LongArray, LongLongArray


cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost

cdef class CarrayContainer:

    def __init__(self, int carray_size=0, dict carrays_to_register=None, **kwargs):
        """Create container of carrays of each of length array_size.

        Parameters
        ----------
        carray_size : int
            Size of each carray in the container.

        carrays_to_register : dict
           Dictionary where the keys are the names and values are the data type of
           carrays to create in the container.

        """
        cdef str carray_name, dtype

        self.carrays = {}
        self.carray_dtypes = {}
        self.carray_named_groups = {}

        if carrays_to_register is not None:
            for carray_name in carrays_to_register:
                dtype = carrays_to_register[carray_name]
                self.register_carray(carray_size, carray_name, dtype)

    cpdef register_carray(self, int carray_size, str carray_name, str dtype="double"):
        """Register new carray in the container.

        Parameters
        ----------
        carray_size : int
            Size of the carray.

        carray_name : str
            Name of carray to be added.

        dtype : str
            Data type of carray.
        """
        if carray_name in self.carrays.keys():
            raise RuntimeError("ERROR: Carray already registered")

        if len(self.carrays) != 0:
            if carray_size != self.get_carray_size():
                raise RuntimeError("ERROR: Size inconsistent with carray size")

        # store data type of field
        self.carray_dtypes[carray_name] = dtype

        if dtype == "double":
            self.carrays[carray_name] = DoubleArray(carray_size)
        elif dtype == "int":
            self.carrays[carray_name] = IntArray(carray_size)
        elif dtype == "long":
            self.carrays[carray_name] = LongArray(carray_size)
        elif dtype == "longlong":
            self.carrays[carray_name] = LongLongArray(carray_size)
        else:
            raise ValueError("ERROR: Unrecognized dtype: %s" % dtype)

    def __getitem__(self, str carray_name):
        """Access carrays as numpy array.

        Parameters
        ----------
        carray_name : str
            Name of carray to retreive.

        Returns
        -------
        numpy array
            Numpy array reference to carray.

        """
        if carray_name in self.carrays.keys():
            return self.carrays[carray_name].get_npy_array()
        else:
            raise AttributeError("Unrecognized field: %s" % carray_name)

    cpdef int get_carray_size(self):
        """Return the number of items in the carray."""
        if len(self.carrays) > 0:
            return self.carrays.values()[0].length
        else:
            return 0

    cpdef extend(self, int increase_carray_size):
        """Increase the total number of items by requested amount.

        Parameters
        ----------
        increase_carray_size : int
            Size to increase container by.

        """
        if increase_carray_size < 0:
            raise RuntimeError("ERROR: Negative number in extend")

        cdef int old_size = self.get_carray_size()
        cdef int new_size = old_size + increase_carray_size
        cdef BaseArray carr

        for carr in self.carrays.values():
            carr.resize(new_size)

    cpdef BaseArray get_carray(self, str carray_name):
        """Return the carray from the container.

        Parameters
        ----------
        carray_name : str
            Name of carray to be returned.

        """
        if PyDict_Contains(self.carrays, carray_name) == 1:
            return <BaseArray> PyDict_GetItem(self.carrays, carray_name)
        else:
            raise KeyError("ERROR: carray %s not present" % carray_name)

    cpdef resize(self, int carray_size):
        """Resize all arrays to the new size.

        Parameters
        ----------
        carray_size : int
            Size of the carray.
        """
        if carray_size < 0:
            raise RuntimeError("ERROR: Negative number in resize")

        cdef BaseArray carr
        for carr in self.carrays.values():
            carr.resize(carray_size)

    def get_sendbufs(self, np.ndarray indices):
        """Slice out values from indices and store it a dictionary of numpy
        arrays.

        Parameters
        ----------
        indices : np.ndarray
            Indices to copy values from.

        Returns
        -------
        dict
            Dictionary of numpy arrays of selected data by indices.
        """
        cdef str carray_name
        cdef dict sendbufs = {}

        for carray_name in self.carrays:
            sendbufs[carray_name] = self[carray_name][indices]
        return sendbufs

    cpdef int append_container(self, CarrayContainer container):
        """Append a container to current container. Carrays that are
        not there in self will not be added.

        Parameters
        ----------
        container : Container
            Container that will be append to self.

        """
        if container.get_carray_size() == 0:
            return 0

        cdef str carray_name
        cdef np.ndarray dst_nparr, src_nparr
        cdef BaseArray dst_carray, src_carray
        cdef int old_num_items = self.get_carray_size()
        cdef int num_extra_items = container.get_carray_size()

        # extend current arrays by the required size 
        self.extend(num_extra_items)

        # should check that fields are equal or not error
        for carray_name in container.carrays.keys():
            if PyDict_Contains(self.carrays, carray_name):
                dst_carray = <BaseArray> PyDict_GetItem(self.carrays, carray_name)
                src_carray = <BaseArray> PyDict_GetItem(container.carrays, carray_name)

                src_nparr = src_carray.get_npy_array()
                dst_nparr = dst_carray.get_npy_array()
                dst_nparr[old_num_items:] = src_nparr

    cpdef CarrayContainer extract_items(self, LongArray index_array, list carray_list_names=None):
        """Create new carray container for item indices in index_array.

        Parameters
        ----------
        index_array : np.ndarray
            Indices of items to be extracted.

        carray_list_names : list
            The list of carrays to extract, if None all carrays
            are extracted.

        """
        cdef str carray_name, dtype
        cdef long size = index_array.length
        cdef BaseArray dst_carray, src_carray
        cdef CarrayContainer result_array = CarrayContainer()

        if carray_list_names is None:
            carray_list_names = self.carrays.keys()

        # now we have the result array setup
        # resize it
        if size == 0:
            return result_array

        # allocate carrays
        for carray_name in carray_list_names:
            dtype = self.carray_dtypes[carray_name]
            result_array.register_carray(size, carray_name, dtype)

        # copy the required indices for each carray
        for carray_name in carray_list_names:
            src_carray = self.get_carray(carray_name)
            dst_carray = result_array.get_carray(carray_name)
            src_carray.copy_values(index_array, dst_carray)

        return result_array

    cpdef remove_items(self, np.ndarray index_list):
        """Remove items whose indices are given in index_list.

        We repeatedly interchange the values of the last element and values from
        the index_list and reduce the size of the array by one. This is done for
        every property and temporary arrays that is being maintained.

        Parameters
        ---------
        index_list : np.ndarray
            array of indices, this array should be a LongArray

        """
        cdef int i
        cdef str msg
        cdef BaseArray carr
        cdef list carray_list
        cdef np.ndarray sorted_indices

        if index_list.size > self.get_carray_size():
            msg = "ERROR: Number of items to be removed is greater than"
            msg += "number of items in array"
            raise ValueError(msg)

        sorted_indices = np.sort(index_list)
        carray_list = self.carrays.values()

        for i in range(len(carray_list)):
            carr = carray_list[i]
            carr.remove(sorted_indices, 1)

    cpdef copy(self, CarrayContainer container, LongArray indices, list carray_list_names):
        """Copy values at indices from container to this container. This is similar to
        paste except the indices are from the source container. Self will be resized
        all contents will be overwritten.

        Parameters
        ----------
        container : CarrayContainer
            Container with values that will be copied to self.

        indices : LongArray
            Indices of container to copy to self.

        """
        cdef str carray_name
        cdef BaseArray dst_array, src_array

        # resize array
        self.resize(indices.length)

        # copy the required indices for each property
        for carray_name in carray_list_names:
            dst_array = self.get_carray(carray_name)
            src_array = container.get_carray(carray_name)
            src_array.copy_values(indices, dst_array)

    cpdef paste(self, CarrayContainer container, LongArray indices, list carray_list_names):
        """Copy values from container to this container at given indices. The
        input agrument should be the same size of indices.

        Parameters
        ----------
        container : CarrayContainer
            Container where values will be taken from.

        indices : LongArray
            Indices to place copy values.

        """
        cdef str carray_name
        cdef BaseArray dst_prop_array, src_prop_array

        if indices.length != container.get_carray_size():
            raise RuntimeError("ERROR: inconsistent carray sizes!")

        # copy the required indices for each property
        for carray_name in carray_list_names:
            dst_array = self.get_carray(carray_name)
            src_array = container.get_carray(carray_name)
            src_array.paste_values(indices, dst_array)

    cpdef add(self, CarrayContainer container, LongArray indices, list carray_list_names):
        """Add values from container to this container at given indices. The
        input agrument should be the same size of indices.

        Parameters
        ----------
        container : CarrayContainer
            Container where values will be taken from.

        indices : LongArray
            Indices to add values to.

        """
        cdef str carray_name
        cdef BaseArray dst_carray, src_carray

        if indices.length != container.get_carray_size():
            raise RuntimeError("ERROR: inconsistent carray sizes!")

        # copy the required indices for each property
        for carray_name in carray_list_names:
            dst_carray = self.get_carray(carray_name)
            src_carray = container.get_carray(carray_name)
            src_carray.add_values(indices, dst_carray)

    cpdef remove_tagged_particles(self, np.int8_t tag):
        """Remove particles that have the given tag.

        Parameters
        ----------

        tag : int8
            The type of particles that need to be removed.

        """
        cdef int i
        cdef LongArray indices = LongArray()
        cdef IntArray tag_array = self.carrays["tag"]
        cdef np.int8_t* tagarrptr = tag_array.get_data_ptr()
        cdef np.ndarray ind

        # find the indices of the particles to be removed
        for i in range(tag_array.length):
            if tagarrptr[i] == tag:
                indices.append(i)

        # remove the particles
        ind = indices.get_npy_array()
        self.remove_items(ind)

    cdef void pointer_groups(self, np.float64_t *vec[], list carray_list_names):
        """Populate a pointer array with references dictated by a list of
        carray names. Only float64_t type are allowed.

        Parameters
        ----------
        vec : *np.float64_t[]
            Float array pointer.

        carray_list_names : list
            List of carray names that vec will point to.
        """
        cdef int i
        cdef str carray_name
        cdef DoubleArray arr

        i = 0
        for carray_name in carray_list_names:
            if carray_name in self.carrays.keys():
                arr = <DoubleArray> self.get_carray(carray_name)
                vec[i] = arr.get_data_ptr()
                i += 1
            else:
                raise ValueError("ERROR: Unknown field!")
