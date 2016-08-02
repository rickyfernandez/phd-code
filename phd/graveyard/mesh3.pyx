import numpy as np
cimport numpy as np

cimport libc.stdlib as stdlib

from utils.particle_tags import ParticleTAGS

from mesh cimport Tess2d, Tess3d
from utils.carray cimport DoubleArray, LongArray, IntArray
from boundary.boundary cimport Boundary
from containers.containers cimport ParticleContainer, CarrayContainer

cdef class PyTess2d(PyTess):
    def __cinit__(self):
        self.thisptr = new Tess2d()

    def __dealloc__(self):
        del self.thisptr

    cdef void reset_tess(self):
        self.thisptr.reset_tess()

    cdef int build_initial_tess(self, double *x[3], double *radius_sq, int num_particles, double huge):
        self.thisptr.build_initial_tess(x, radius_sq, num_particles, huge)

    cdef int update_initial_tess(self, double *x[3], int up_num_particles):
        self.thisptr.update_initial_tess(x, up_num_particles)

    cdef int count_number_of_faces(self):
        self.thisptr.count_number_of_faces()

    cdef int extract_geometry(self, double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j):
        self.thisptr.extract_geometry(x, dcenter_of_mass, volume,
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
        self.thisptr.build_initial_tess(x, radius_sq, num_particles, huge)

    cdef int update_initial_tess(self, double *x[3], int up_num_particles):
        self.thisptr.update_initial_tess(x, up_num_particles)

    cdef int count_number_of_faces(self):
        self.thisptr.count_number_of_faces()

    cdef int extract_geometry(self, double* x[3], double* dcenter_of_mass[3], double* volume,
                double* face_area, double* face_com[3], double* face_n[3],
                int* pair_i, int* pair_j):
        self.thisptr.extract_geometry(x, dcenter_of_mass, volume,
                face_area, face_com, face_n,
                pair_i, pair_j)

cdef class Mesh:
    def __init__(self, Boundary boundary):
        cdef dim = boundary.domain.dim

        self.boundary = boundary

        face_vars = {
                "area": "double",
                "pair-i": "long",
                "pair-j": "long"
                }

        for axis in "xyz"[:dim]:
            face_vars["velocity-" + axis] = "double"
            face_vars["normal-" + axis] = "double"
            face_vars["com-" + axis] = "double"

        self.faces = ParticleContainer(var_dict=face_vars)

        if dim == 2:
            self.tess = Tess2d()
        elif dim == 3:
            self.tess = Tess3d()

    def tessellate(self, ParticleContainer pc):
        self._tessellate(pc)

    cdef _tessellate(self, ParticleContainer pc):

        cdef DoubleArray r = pc.get_carray("radius")
        cdef np.float64_t *xp[3]
        cdef np.float64_t *rp

        cdef int fail

        # initial mesh should only have local particles
        pc.remove_tagged_particles(ParticleTAGS.Ghost)
        pc.extract_field_vec_ptr(xp, "position")
        rp = r.get_data_ptr()

        # add local particles to the tessellation
        fail = self.tess.build_initial_tess(xp, rp, pc.get_number_of_particles())
        assert(fail != -1)

        # pc modified - ghost particles append to container
        num_ghost = self.boundary._create_ghost_particles(pc)

        # creating ghost may have called remalloc
        pc.extract_field_vec_ptr(xp, "position")
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

        cdef int num_faces, i, j, fail
        cdef list fields = ["volume"]

        # release memory used in the tessellation
        self.reset_mesh()
        self._tessellate()

        # allocate memory for face information
        num_faces = self.tess.count_number_of_faces()
        self.faces.resize(num_faces)

        # pointers to particle data 
        pc.extract_field_vec_ptr(x, "position")
        pc.extract_field_vec_ptr(dcom, "dcom")
        vol = p_vol.get_data_ptr()

        # pointers to face data
        faces.extract_field_vec_ptr(nx, "normal")
        faces.extract_field_vec_ptr(com, "com")
        pair_i = f_pair_i.get_data_ptr()
        pair_j = f_pair_j.get_data_ptr()
        area   = f_area.get_data_ptr()

        # store particle and face information for the tessellation
        # only real particle information is computed
        fail = self.tess.extract_geometry(x, dcom, vol,
                area, com, nx, <int*>pair_ip, <int*>pair_jp)
        assert(fail != -1)

        # transfer particle information to ghost particles
        for axis in "xyz"[:dim]:
            fields.append("dcom-" + axis)
        self.boundary._update_ghost_particles(pc, fields)

    def reset_mesh(self):
        self.tess.reset_tess()


#cdef class MeshBase:
#
#    cdef _tessellate(self):
#        msg = "MeshBase::_tessellate called!"
#        raise NotImplementedError(msg)
#
#    cdef _build_geometry(self):
#        msg = "MeshBase::_build_geometry called!"
#        raise NotImplementedError(msg)
#cdef class Mesh2d(MeshBase):
#
#    def __cinit__(self, ParticleContainer pc, Boundary boundary):
#
#        self.dim = 2
#        self.tess = Tess2d()
#        self.particles = pc
#        self.boundary = boundary
#
#        face_vars = {
#                "area": "double",
#                "velocity-x": "double",
#                "velocity-y": "double",
#                "normal-x": "double",
#                "normal-y": "double",
#                "com-x": "double",
#                "com-y": "double",
#                "pair-i": "long",
#                "pair-j": "long"
#                }
#        self.faces = ParticleContainer(var_dict=face_vars)
#
#    def tessellate(self):
#        self._tessellate()
#
#    cdef _tessellate(self):
#
#        cdef ParticleContainer pc = self.particles
#
#        # particle information
#        cdef DoubleArray x = pc.get_carray("position-x")
#        cdef DoubleArray y = pc.get_carray("position-y")
#        cdef DoubleArray r = pc.get_carray("radius")
#
#        cdef np.float64_t* xp = x.get_data_ptr()
#        cdef np.float64_t* yp = y.get_data_ptr()
#        cdef np.float64_t* rp = r.get_data_ptr()
#
#        cdef int fail
#
#        pc.remove_tagged_particles(Ghost)
#        fail = self.tess.build_initial_tess(xp, yp, rp, pc.get_number_of_particles())
#        assert(fail != -1)
#
#        # the boundary should become parallel
#        num_ghost = self.boundary._create_ghost_particles(pc)
#
#        # creating ghost may have remalloc
#        xp = x.get_data_ptr(); yp = y.get_data_ptr()
#        self.tess.update_initial_tess(xp, yp, num_ghost);
#
#    def build_geometry(self):
#        self._build_geometry()
#
#    cdef _build_geometry(self):
#        # note should make function that returns void pointer
#        cdef ParticleContainer pc = self.particles
#
#        cdef DoubleArray x = pc.get_carray("position-x")
#        cdef DoubleArray y = pc.get_carray("position-y")
#        cdef DoubleArray pcom_x = pc.get_carray("com-x")
#        cdef DoubleArray pcom_y = pc.get_carray("com-y")
#
#        cdef DoubleArray rho = pc.get_carray("density")
#        cdef DoubleArray vol = pc.get_carray("volume")
#        cdef DoubleArray pre = pc.get_carray("pressure")
#
#        cdef LongArray maps = pc.get_carray("map")
#        cdef IntArray tags = pc.get_carray("tag")
#
#        # face information
#        cdef DoubleArray area = self.faces.get_carray("area")
#        cdef DoubleArray com_x = self.faces.get_carray("com-x")
#        cdef DoubleArray com_y = self.faces.get_carray("com-y")
#        cdef DoubleArray n_x = self.faces.get_carray("normal-x")
#        cdef DoubleArray n_y = self.faces.get_carray("normal-y")
#        cdef LongArray pair_i = self.faces.get_carray("pair-i")
#        cdef LongArray pair_j = self.faces.get_carray("pair-j")
#
#        cdef np.float64_t *xp, *yp, *pcom_xp, *pcom_yp, *volp
#
#        cdef np.float64_t *areap, *n_xp, *n_yp, *com_xp, *com_yp
#        cdef np.int32_t *pair_ip, *pair_jp
#
#        cdef int num_faces, i, j, fail
#
#        self.reset_mesh()
#        self._tessellate()
#
#        num_faces = self.tess.count_number_of_faces()
#        self.faces.resize(num_faces)
#
#        # pointers to particle data 
#        xp = x.get_data_ptr(); yp = y.get_data_ptr()
#        pcom_xp = pcom_x.get_data_ptr(); pcom_yp = pcom_y.get_data_ptr()
#        volp = vol.get_data_ptr()
#
#        # pointers to face data
#        n_xp = n_x.get_data_ptr(); n_yp = n_y.get_data_ptr()
#        com_xp = com_x.get_data_ptr(); com_yp = com_y.get_data_ptr()
#        pair_ip = pair_i.get_data_ptr(); pair_jp = pair_j.get_data_ptr()
#        areap = area.get_data_ptr()
#
#        fail = self.tess.extract_geometry(xp, yp, pcom_xp, pcom_yp,
#                volp, areap, com_xp, com_yp, n_xp, n_yp,
#                <int*>pair_ip, <int*>pair_jp)
#        assert(fail != -1)
#
#        print 'building mesh done'
#        #self.boundary._update_ghost_particles(pc)
#        #for i in range(self.particles.get_number_of_particles()):
#        #    if tags[i] == Ghost:
##
##                j = maps[i]
##                vol.data[i] = vol.data[j]
#
#    def reset_mesh(self):
#        self.tess.reset_tess()
#
#cdef class Mesh3d(MeshBase):
#
#    def __cinit__(self, ParticleContainer pc, Boundary boundary):
#
#        self.dim = 3
#        self.tess = Tess3d()
#        self.particles = pc
#        self.boundary = boundary
#
#        face_vars = {
#                "area": "double",
#                "velocity-x": "double",
#                "velocity-y": "double",
#                "velocity-z": "double",
#                "normal-x": "double",
#                "normal-y": "double",
#                "normal-z": "double",
#                "com-x": "double",
#                "com-y": "double",
#                "com-z": "double",
#                "pair-i": "long",
#                "pair-j": "long"
#                }
#        self.faces = ParticleContainer(var_dict=face_vars)
#
#    def tessellate(self):
#        self._tessellate()
#
#    cdef _tessellate(self):
#
#        cdef ParticleContainer pc = self.particles
#
#        # particle information
#        cdef DoubleArray x = pc.get_carray("position-x")
#        cdef DoubleArray y = pc.get_carray("position-y")
#        cdef DoubleArray z = pc.get_carray("position-z")
#        cdef DoubleArray r = pc.get_carray("radius")
#
#        cdef np.float64_t* xp = x.get_data_ptr()
#        cdef np.float64_t* yp = y.get_data_ptr()
#        cdef np.float64_t* zp = z.get_data_ptr()
#        cdef np.float64_t* rp = r.get_data_ptr()
#
#        cdef int fail
#
#        pc.remove_tagged_particles(Ghost)
#        fail = self.tess.build_initial_tess(xp, yp, zp, rp, pc.get_number_of_particles(), 1.0E30)
#        assert(fail != -1)
#
#        # the boundary should become parallel
#        num_ghost = self.boundary._create_ghost_particles(pc)
#
#        # creating ghost may have remalloc
#        xp = x.get_data_ptr(); yp = y.get_data_ptr(); zp = z.get_data_ptr();
#        self.tess.update_initial_tess(xp, yp, zp, num_ghost);
#
#    def build_geometry(self):
#        # note should make function that returns void pointer
#        cdef ParticleContainer pc = self.particles
#
#        cdef DoubleArray x = pc.get_carray("position-x")
#        cdef DoubleArray y = pc.get_carray("position-y")
#        cdef DoubleArray z = pc.get_carray("position-z")
#        cdef DoubleArray pcom_x = pc.get_carray("com-x")
#        cdef DoubleArray pcom_y = pc.get_carray("com-y")
#        cdef DoubleArray pcom_z = pc.get_carray("com-z")
#
#        cdef DoubleArray rho = pc.get_carray("density")
#        cdef DoubleArray vol = pc.get_carray("volume")
#        cdef DoubleArray pre = pc.get_carray("pressure")
#
#        cdef LongArray maps = pc.get_carray("map")
#        cdef IntArray tags = pc.get_carray("tag")
#
#        # face information
#        cdef DoubleArray area = self.faces.get_carray("area")
#        cdef DoubleArray com_x = self.faces.get_carray("com-x")
#        cdef DoubleArray com_y = self.faces.get_carray("com-y")
#        cdef DoubleArray com_z = self.faces.get_carray("com-z")
#        cdef DoubleArray n_x = self.faces.get_carray("normal-x")
#        cdef DoubleArray n_y = self.faces.get_carray("normal-y")
#        cdef DoubleArray n_z = self.faces.get_carray("normal-z")
#        cdef LongArray pair_i = self.faces.get_carray("pair-i")
#        cdef LongArray pair_j = self.faces.get_carray("pair-j")
#
#        cdef np.float64_t *xp, *yp, *zp, *pcom_xp, *pcom_yp, *pcom_zp, *volp
#
#        cdef np.float64_t *areap, *n_xp, *n_yp, *n_zp, *com_xp, *com_yp, *com_zp
#        cdef np.int32_t *pair_ip, *pair_jp
#
#        cdef int num_faces, i, j, fail
#
#
#        self.reset_mesh()
#        self.tessellate()
#
#        num_faces = self.tess.count_number_of_faces()
#        self.faces.resize(num_faces)
#
#        # pointers to particle data 
#        xp = x.get_data_ptr(); yp = y.get_data_ptr(); zp = z.get_data_ptr()
#        pcom_xp = pcom_x.get_data_ptr(); pcom_yp = pcom_y.get_data_ptr(); pcom_zp = pcom_z.get_data_ptr()
#        volp = vol.get_data_ptr()
#
#        # pointers to face data
#        n_xp = n_x.get_data_ptr(); n_yp = n_y.get_data_ptr(); n_zp = n_z.get_data_ptr()
#        com_xp = com_x.get_data_ptr(); com_yp = com_y.get_data_ptr(); com_zp = com_z.get_data_ptr()
#        pair_ip = pair_i.get_data_ptr(); pair_jp = pair_j.get_data_ptr()
#        areap = area.get_data_ptr()
#
#        fail = self.tess.extract_geometry(xp, yp, zp, pcom_xp, pcom_yp, pcom_zp,
#                volp, areap, com_xp, com_yp, com_zp, n_xp, n_yp, n_zp,
#                <int*>pair_ip, <int*>pair_jp)
#        assert(fail != -1)
#
##        self.faces.resize(fail)
##        self.boundary._update_ghost_particles(pc)
#
#        #for i in range(self.particles.get_number_of_particles()):
#        #    if tags[i] == Ghost:
##
##                j = maps[i]
##                vol.data[i] = vol.data[j]
#
#    def reset_mesh(self):
#        self.tess.reset_tess()
