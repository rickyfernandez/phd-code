import numpy as np
cimport numpy as np

cimport libc.stdlib as stdlib

from utils.particle_tags import ParticleTAGS

from mesh cimport Tess2d, Tess3d
from utils.carray cimport DoubleArray, LongArray, IntArray
from boundary.boundary cimport BoundaryParallelBase
from containers.containers cimport ParticleContainer, CarrayContainer

cdef int Ghost = ParticleTAGS.Ghost

cdef class MeshBase:

    cdef _tessellate(self):
        msg = "MeshBase::_tessellate called!"
        raise NotImplementedError(msg)

    cdef _build_geometry(self):
        msg = "MeshBase::_build_geometry called!"
        raise NotImplementedError(msg)

cdef class Mesh2d(MeshBase):

    def __cinit__(self, ParticleContainer pc, BoundaryParallelBase boundary):

        self.dim = 2
        self.tess = Tess2d()
        self.particles = pc
        self.boundary = boundary

        face_vars = {
                "area": "double",
                "velocity-x": "double",
                "velocity-y": "double",
                "normal-x": "double",
                "normal-y": "double",
                "com-x": "double",
                "com-y": "double",
                "pair-i": "long",
                "pair-j": "long"
                }
        self.faces = ParticleContainer(var_dict=face_vars)

    def tessellate(self):
        self._tessellate()

    cdef _tessellate(self):

        cdef ParticleContainer pc = self.particles

        # particle information
        cdef DoubleArray x = pc.get_carray("position-x")
        cdef DoubleArray y = pc.get_carray("position-y")
        cdef DoubleArray r = pc.get_carray("radius")

        cdef np.float64_t* xp = x.get_data_ptr()
        cdef np.float64_t* yp = y.get_data_ptr()
        cdef np.float64_t* rp = r.get_data_ptr()

        cdef int fail

        pc.remove_tagged_particles(Ghost)
        fail = self.tess.build_initial_tess(xp, yp, rp, pc.get_number_of_particles())
        assert(fail != -1)

        # the boundary should become parallel
        num_ghost = self.boundary._create_ghost_particles(pc)

        # creating ghost may have remalloc
        xp = x.get_data_ptr(); yp = y.get_data_ptr()
        self.tess.update_initial_tess(xp, yp, num_ghost);

    def build_geometry(self):
        self._build_geometry()

    cdef _build_geometry(self):
        # note should make function that returns void pointer
        cdef ParticleContainer pc = self.particles

        cdef DoubleArray x = pc.get_carray("position-x")
        cdef DoubleArray y = pc.get_carray("position-y")
        cdef DoubleArray pcom_x = pc.get_carray("com-x")
        cdef DoubleArray pcom_y = pc.get_carray("com-y")

        cdef DoubleArray rho = pc.get_carray("density")
        cdef DoubleArray vol = pc.get_carray("volume")
        cdef DoubleArray pre = pc.get_carray("pressure")

        cdef LongArray maps = pc.get_carray("map")
        cdef IntArray tags = pc.get_carray("tag")

        # face information
        cdef DoubleArray area = self.faces.get_carray("area")
        cdef DoubleArray com_x = self.faces.get_carray("com-x")
        cdef DoubleArray com_y = self.faces.get_carray("com-y")
        cdef DoubleArray n_x = self.faces.get_carray("normal-x")
        cdef DoubleArray n_y = self.faces.get_carray("normal-y")
        cdef LongArray pair_i = self.faces.get_carray("pair-i")
        cdef LongArray pair_j = self.faces.get_carray("pair-j")

        cdef np.float64_t *xp, *yp, *pcom_xp, *pcom_yp, *volp

        cdef np.float64_t *areap, *n_xp, *n_yp, *com_xp, *com_yp
        cdef np.int32_t *pair_ip, *pair_jp

        cdef int num_faces, i, j, fail

        self.reset_mesh()
        self._tessellate()

        num_faces = self.tess.count_number_of_faces()
        self.faces.resize(num_faces)

        # pointers to particle data 
        xp = x.get_data_ptr(); yp = y.get_data_ptr()
        pcom_xp = pcom_x.get_data_ptr(); pcom_yp = pcom_y.get_data_ptr()
        volp = vol.get_data_ptr()

        # pointers to face data
        n_xp = n_x.get_data_ptr(); n_yp = n_y.get_data_ptr()
        com_xp = com_x.get_data_ptr(); com_yp = com_y.get_data_ptr()
        pair_ip = pair_i.get_data_ptr(); pair_jp = pair_j.get_data_ptr()
        areap = area.get_data_ptr()

        fail = self.tess.extract_geometry(xp, yp, pcom_xp, pcom_yp,
                volp, areap, com_xp, com_yp, n_xp, n_yp,
                <int*>pair_ip, <int*>pair_jp)
        assert(fail != -1)

        #self.boundary._update_ghost_particles(pc)
        #for i in range(self.particles.get_number_of_particles()):
        #    if tags[i] == Ghost:
#
#                j = maps[i]
#                vol.data[i] = vol.data[j]

    def reset_mesh(self):
        self.tess.reset_tess()

cdef class Mesh3d(MeshBase):

    def __cinit__(self, ParticleContainer pc, BoundaryParallelBase boundary):

        self.dim = 3
        self.tess = Tess3d()
        self.particles = pc
        self.boundary = boundary

        face_vars = {
                "area": "double",
                "velocity-x": "double",
                "velocity-y": "double",
                "velocity-z": "double",
                "normal-x": "double",
                "normal-y": "double",
                "normal-z": "double",
                "com-x": "double",
                "com-y": "double",
                "com-z": "double",
                "pair-i": "long",
                "pair-j": "long"
                }
        self.faces = ParticleContainer(var_dict=face_vars)

    def tessellate(self):
        self._tessellate()

    cdef _tessellate(self):

        cdef ParticleContainer pc = self.particles

        # particle information
        cdef DoubleArray x = pc.get_carray("position-x")
        cdef DoubleArray y = pc.get_carray("position-y")
        cdef DoubleArray z = pc.get_carray("position-z")
        cdef DoubleArray r = pc.get_carray("radius")

        cdef np.float64_t* xp = x.get_data_ptr()
        cdef np.float64_t* yp = y.get_data_ptr()
        cdef np.float64_t* zp = z.get_data_ptr()
        cdef np.float64_t* rp = r.get_data_ptr()

        cdef int fail

        pc.remove_tagged_particles(Ghost)
        fail = self.tess.build_initial_tess(xp, yp, zp, rp, pc.get_number_of_particles(), 1.0E30)
        assert(fail != -1)

        # the boundary should become parallel
        num_ghost = self.boundary._create_ghost_particles(pc)

        # creating ghost may have remalloc
        xp = x.get_data_ptr(); yp = y.get_data_ptr(); zp = z.get_data_ptr();
        self.tess.update_initial_tess(xp, yp, zp, num_ghost);

    def build_geometry(self):
        # note should make function that returns void pointer
        cdef ParticleContainer pc = self.particles

        cdef DoubleArray x = pc.get_carray("position-x")
        cdef DoubleArray y = pc.get_carray("position-y")
        cdef DoubleArray z = pc.get_carray("position-z")
        cdef DoubleArray pcom_x = pc.get_carray("com-x")
        cdef DoubleArray pcom_y = pc.get_carray("com-y")
        cdef DoubleArray pcom_z = pc.get_carray("com-z")

        cdef DoubleArray rho = pc.get_carray("density")
        cdef DoubleArray vol = pc.get_carray("volume")
        cdef DoubleArray pre = pc.get_carray("pressure")

        cdef LongArray maps = pc.get_carray("map")
        cdef IntArray tags = pc.get_carray("tag")

        # face information
        cdef DoubleArray area = self.faces.get_carray("area")
        cdef DoubleArray com_x = self.faces.get_carray("com-x")
        cdef DoubleArray com_y = self.faces.get_carray("com-y")
        cdef DoubleArray com_z = self.faces.get_carray("com-z")
        cdef DoubleArray n_x = self.faces.get_carray("normal-x")
        cdef DoubleArray n_y = self.faces.get_carray("normal-y")
        cdef DoubleArray n_z = self.faces.get_carray("normal-z")
        cdef LongArray pair_i = self.faces.get_carray("pair-i")
        cdef LongArray pair_j = self.faces.get_carray("pair-j")

        cdef np.float64_t *xp, *yp, *zp, *pcom_xp, *pcom_yp, *pcom_zp, *volp

        cdef np.float64_t *areap, *n_xp, *n_yp, *n_zp, *com_xp, *com_yp, *com_zp
        cdef np.int32_t *pair_ip, *pair_jp

        cdef int num_faces, i, j, fail


        self.reset_mesh()
        self.tessellate()

        num_faces = self.tess.count_number_of_faces()
        self.faces.resize(num_faces)

        # pointers to particle data 
        xp = x.get_data_ptr(); yp = y.get_data_ptr(); zp = z.get_data_ptr()
        pcom_xp = pcom_x.get_data_ptr(); pcom_yp = pcom_y.get_data_ptr(); pcom_zp = pcom_z.get_data_ptr()
        volp = vol.get_data_ptr()

        # pointers to face data
        n_xp = n_x.get_data_ptr(); n_yp = n_y.get_data_ptr(); n_zp = n_z.get_data_ptr()
        com_xp = com_x.get_data_ptr(); com_yp = com_y.get_data_ptr(); com_zp = com_z.get_data_ptr()
        pair_ip = pair_i.get_data_ptr(); pair_jp = pair_j.get_data_ptr()
        areap = area.get_data_ptr()

        fail = self.tess.extract_geometry(xp, yp, zp, pcom_xp, pcom_yp, pcom_zp,
                volp, areap, com_xp, com_yp, com_zp, n_xp, n_yp, n_zp,
                <int*>pair_ip, <int*>pair_jp)
        assert(fail != -1)

        self.faces.resize(fail)
        self.boundary._update_ghost_particles(pc)

        #for i in range(self.particles.get_number_of_particles()):
        #    if tags[i] == Ghost:
#
#                j = maps[i]
#                vol.data[i] = vol.data[j]

    def reset_mesh(self):
        self.tess.reset_tess()
