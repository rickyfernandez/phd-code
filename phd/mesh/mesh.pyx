import numpy as np
cimport numpy as np
cimport libc.stdlib as stdlib

from ..boundary.boundary cimport Boundary
from ..utils.particle_tags import ParticleTAGS
from ..utils.carray cimport DoubleArray, LongArray, IntArray
from ..containers.containers cimport ParticleContainer, CarrayContainer

cdef class PyTess:
    cdef void reset_tess(self):
        raise NotImplementedError, 'PyTess::reset_tess'

    cdef int build_initial_tess(self, double *x[3], double *radius_sq, int num_particles, double huge):
        raise NotImplementedError, 'PyTess::build_initial_tess'

    cdef int update_initial_tess(self, double *x[3], int up_num_particles):
        raise NotImplementedError, 'PyTess::update_initial_tess'

    cdef int count_number_of_faces(self):
        raise NotImplementedError, 'PyTess::count_number_of_faces'

    cdef int extract_geometry(self, double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j):
        raise NotImplementedError, 'PyTess::extract_geometry'

cdef class PyTess2d(PyTess):
    def __cinit__(self):
        self.thisptr = new Tess2d()

    def __dealloc__(self):
        del self.thisptr

    cdef void reset_tess(self):
        self.thisptr.reset_tess()

    cdef int build_initial_tess(self, double *x[3], double *radius_sq, int num_particles, double huge):
        return self.thisptr.build_initial_tess(x, radius_sq, num_particles, huge)

    cdef int update_initial_tess(self, double *x[3], int up_num_particles):
        return self.thisptr.update_initial_tess(x, up_num_particles)

    cdef int count_number_of_faces(self):
        return self.thisptr.count_number_of_faces()

    cdef int extract_geometry(self, double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j):
        return self.thisptr.extract_geometry(x, dcenter_of_mass, volume,
                face_area, face_com, face_n,
                pair_i, pair_j)

cdef class PyTess3d(PyTess):
    def __cinit__(self):
        self.thisptr = new Tess3d()

    def __dealloc__(self):
        del self.thisptr

    cdef void reset_tess(self):
        self.thisptr.reset_tess()

    cdef int build_initial_tess(self, double *x[3], double *radius_sq, int num_particles, double huge):
        return self.thisptr.build_initial_tess(x, radius_sq, num_particles, huge)

    cdef int update_initial_tess(self, double *x[3], int up_num_particles):
        return self.thisptr.update_initial_tess(x, up_num_particles)

    cdef int count_number_of_faces(self):
        return self.thisptr.count_number_of_faces()

    cdef int extract_geometry(self, double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j):
        return self.thisptr.extract_geometry(x, dcenter_of_mass, volume,
                face_area, face_com, face_n,
                pair_i, pair_j)

cdef class Mesh:
    def __init__(self, Boundary boundary):
        cdef dim = boundary.domain.dim

        self.boundary = boundary
        self.dim = dim

        #self.neighbors = nns_vec(128, nns)

        face_vars = {
                "area": "double",
                "pair-i": "long",
                "pair-j": "long",
                "com-x": "double",
                "com-y": "double",
                "velocity-x": "double",
                "velocity-y": "double",
                "normal-x": "double",
                "normal-y": "double",
                }

        # which fields to upate in pc
        self.fields = [
                "volume",
                "dcom-x",
                "dcom-y"
                ]

        #for axis in "xyz"[:dim]:
        #    self.fields.append("dcom-" + axis)
            #face_vars["velocity-" + axis] = "double"
            #face_vars["normal-" + axis] = "double"
            #face_vars["com-" + axis] = "double"

        self.faces = CarrayContainer(var_dict=face_vars)
        self.faces.named_groups['velocity'] = ['velocity-x', 'velocity-y']
        self.faces.named_groups['normal'] = ['normal-x', 'normal-y']
        self.faces.named_groups['com'] = ['com-x', 'com-y']

        if dim == 2:
            self.tess = PyTess2d()
        elif dim == 3:
            self.tess = PyTess3d()
            self.fields.append("dcom-z")
            self.faces.named_groups['velocity'].append('velocity-z')
            self.faces.named_groups['normal'].append('normal-z')
            self.faces.named_groups['com'].append('com-z')

    def tessellate(self, ParticleContainer pc):
        self._tessellate(pc)

    cdef _tessellate(self, ParticleContainer pc):

        cdef DoubleArray r = pc.get_carray("radius")
        cdef np.float64_t *xp[3]
        cdef np.float64_t *rp

        cdef int fail

        # initial mesh should only have local particles
        pc.remove_tagged_particles(ParticleTAGS.Ghost)
        #pc.extract_field_vec_ptr(xp, "position")
        pc.pointer_groups(xp, pc.named_groups["position"])
        rp = r.get_data_ptr()

        # add local particles to the tessellation
        fail = self.tess.build_initial_tess(xp, rp, pc.get_number_of_particles(), 1.0E33)
        assert(fail != -1)

        # pc modified - ghost particles append to container
        num_ghost = self.boundary._create_ghost_particles(pc)

        # creating ghost may have called remalloc
        #pc.extract_field_vec_ptr(xp, "position")
        pc.pointer_groups(xp, pc.named_groups['position'])
        self.tess.update_initial_tess(xp, num_ghost)

    def build_geometry(self, ParticleContainer pc):
        self._build_geometry(pc)

    cdef _build_geometry(self, ParticleContainer pc):

        # particle information
        cdef DoubleArray p_vol = pc.get_carray("volume")

        # face information
        cdef DoubleArray f_area = self.faces.get_carray("area")
        cdef LongArray f_pair_i = self.faces.get_carray("pair-i")
        cdef LongArray f_pair_j = self.faces.get_carray("pair-j")

        # particle pointers
        cdef np.float64_t *x[3], *dcom[3], *vol

        # face pointers
        cdef np.float64_t *area, *nx[3], *com[3]
        cdef np.int32_t *pair_i, *pair_j

        cdef int num_faces, i, j, fail, dim = self.dim

        # release memory used in the tessellation
        self._reset_mesh()
        self._tessellate(pc)

        # allocate memory for face information
        num_faces = self.tess.count_number_of_faces()
        self.faces.resize(num_faces)

        # pointers to particle data 
        #pc.extract_field_vec_ptr(x, "position")
        #pc.extract_field_vec_ptr(dcom, "dcom")
        pc.pointer_groups(x, pc.named_groups['position'])
        pc.pointer_groups(dcom, pc.named_groups['dcom'])
        vol = p_vol.get_data_ptr()

        # pointers to face data
        #self.faces.extract_field_vec_ptr(nx, "normal")
        #self.faces.extract_field_vec_ptr(com, "com")
        self.faces.pointer_groups(nx,  self.faces.named_groups['normal'])
        self.faces.pointer_groups(com, self.faces.named_groups['com'])
        pair_i = f_pair_i.get_data_ptr()
        pair_j = f_pair_j.get_data_ptr()
        area   = f_area.get_data_ptr()

        #self.neighbors.resize(self.domain.num_real_particles)
        #for i in range(self.domain.num_real_particles):
        #    self.neighbors[i].resize(0)

        # store particle and face information for the tessellation
        # only real particle information is computed
        fail = self.tess.extract_geometry(x, dcom, vol,
                area, com, nx, <int*>pair_i, <int*>pair_j)
        #        self.neighbors)
        assert(fail != -1)

        # tmp for now
        self.faces.resize(fail)

        # transfer particle information to ghost particles
        self.boundary._update_ghost_particles(pc, self.fields)

    def reset_mesh(self):
        self.tess._reset_tess()

    cdef _reset_mesh(self):
        self.tess.reset_tess()
