
cdef class PyTess:
    cdef void reset_tess(self):
        raise NotImplementedError, 'PyTess::reset_tess'

    #cdef int build_initial_tess(self, double *x[3], double *radius_sq, int num_particles):
    cdef int build_initial_tess(self, double *x[3], double *radius_sq, int start_new_ghost, int stop_new_ghost):
        raise NotImplementedError, 'PyTess::build_initial_tess'

    cdef int update_initial_tess(self, double *x[3], int begin_particles, int end_particles):
        raise NotImplementedError, 'PyTess::update_initial_tess'

    cdef int count_number_of_faces(self):
        raise NotImplementedError, 'PyTess::count_number_of_faces'

    cdef int extract_geometry(self, double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j, nn_vec &neighbors):
        raise NotImplementedError, 'PyTess::extract_geometry'

    cdef int update_radius(self, double *x[3], double *radius, cpplist[FlagParticle] flagged_particles):
        pass

cdef class PyTess2d(PyTess):
    def __cinit__(self):
        self.thisptr = new Tess2d()

    def __dealloc__(self):
        del self.thisptr

    cdef void reset_tess(self):
        self.thisptr.reset_tess()

    #cdef int build_initial_tess(self, double *x[3], double *radius_sq, int num_particles):
    cdef int build_initial_tess(self, double *x[3], double *radius_sq, int start_new_ghost, int stop_new_ghost):
        #return self.thisptr.build_initial_tess(x, radius_sq, num_particles)
        return self.thisptr.build_initial_tess(x, radius_sq, start_new_ghost, stop_new_ghost)

    cdef int update_initial_tess(self, double *x[3], int begin_particles, int end_particles):
        return self.thisptr.update_initial_tess(x, begin_particles, end_particles)

    cdef int count_number_of_faces(self):
        return self.thisptr.count_number_of_faces()

    cdef int extract_geometry(self, double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j, nn_vec &neighbors):
        return self.thisptr.extract_geometry(x, dcenter_of_mass, volume,
                face_area, face_com, face_n,
                pair_i, pair_j, neighbors)

    cdef int update_radius(self, double *x[3], double *radius, cpplist[FlagParticle] flagged_particles):
        return self.thisptr.update_radius(x, radius, flagged_particles)

#cdef class PyTess3d(PyTess):
#    def __cinit__(self):
#        self.thisptr = new Tess3d()
#
#    def __dealloc__(self):
#        del self.thisptr
#
#    cdef void reset_tess(self):
#        self.thisptr.reset_tess()
#
#    cdef int build_initial_tess(self, double *x[3], double *radius_sq, int num_particles, double huge):
#        return self.thisptr.build_initial_tess(x, radius_sq, num_particles, huge)
#
#    cdef int update_initial_tess(self, double *x[3], int up_num_particles):
#        return self.thisptr.update_initial_tess(x, up_num_particles)
#
#    cdef int count_number_of_faces(self):
#        return self.thisptr.count_number_of_faces()
#
#    cdef int extract_geometry(self, double* x[3], double* dcenter_of_mass[3], double* volume,
#                double* face_area, double* face_com[3], double* face_n[3],
#                int* pair_i, int* pair_j, nn_vec &neighbors):
#        return self.thisptr.extract_geometry(x, dcenter_of_mass, volume,
#                face_area, face_com, face_n,
#                pair_i, pair_j, neighbors)
