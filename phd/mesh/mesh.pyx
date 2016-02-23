import numpy as np
cimport numpy as np

from utils.particle_tags import ParticleTAGS

from tess cimport Tess2d
from utils.carray cimport DoubleArray, LongArray
from boundary.boundary cimport BoundaryBase2d
from containers.containers cimport ParticleContainer, CarrayContainer

cdef int Ghost = ParticleTAGS.Ghost

cdef class Mesh2d:

    def __cinit__(ParticleContainer pc, BoundaryBase2d boundary):

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

    def tessellate():

        # particle information
        cdef DoubleArray x = self.pc.get_carray("position-x")
        cdef DoubleArray y = self.pc.get_carray("position-y")
        cdef DoubleArray r = self.pc.get_carray("radius")

        cdef np.float64_t* xp = x.get_data_ptr()
        cdef np.float64_t* yp = y.get_data_ptr()
        cdef np.float64_t* rp = r.get_data_ptr()

        cdef int fail

        fail = self.tess.build_initial_tess(xp, yp, rp, pc.get_number_of_particles())
        assert(fail != -1)

        # the boundary should become parallel
        num_ghost = self.boundary._create_ghost_particles(self.particles)

        # creating ghost may have remalloc
        xp = x.get_data_ptr(); yp = y.get_data_ptr()
        self.tess.update_initial_mesh(xp, yp, num_ghost);

    def buld_geometry():
        # note should make function that returns void pointer

        cdef DoubleArray x = self.pc.get_carray("position-x")
        cdef DoubleArray y = self.pc.get_carray("position-y")
        cdef DoubleArray pcom_x = self.pc.get_carray("com-x")
        cdef DoubleArray pcom_y = self.pc.get_carray("com-y")
        cdef DoubleArray vol = self.pc.get_carray("volume")

        # face information
        cdef DoubleArray area = self.faces.get_carray("area")
        cdef DoubleArray n_x = self.faces.get_carray("normal-x")
        cdef DoubleArray n_y = self.faces.get_carray("normal-y")
        cdef LongArray pair_i = self.faces.get_carray("pair-i")
        cdef LongArray pair_j = self.faces.get_carray("pair-j")

        cdef np.float64_t *xp, *yp, *pcom_xp, *pcom_yp, *volp

        cdef np.float64_t *areap, *n_xp, *n_yp
        cdef np.int32_t *pair_ip, *pair_jp

        cdef int num_faces, i, j, fail

        self.tessellate()

        num_faces = self.tess.count_number_of_faces()
        faces.resize(num_faces)

        # pointers to particle data 
        xp = x.get_data_ptr(); yp = y.get_data_ptr()
        pcom_xp = pcom_x.get_data_ptr(); pcom_yp = pcom_y.get_data_ptr()
        volp = vol.get_data_ptr()

        # pointers to face data
        n_xp = n_x.get_data_ptr(); n_yp = n_y.get_data_ptr()
        pair_ip = pair_i.get_data_ptr(); pair_jp = pair_j.get_data_ptr()
        areap = area.get_data_ptr()

        fail = self.tess.extract_geometry(xp, yp, pcom_xp, pcom_yp,
                volp, areap, n_xp, n_yp, <int*>pair_ip, <int*>pair_jp)
        assert(fail != -1)

        for i in range(self.particles.get_number_of_particles()):
            if tag[i] == Ghost:

                j = map[i]
                vol.data[i] = vol.data[j]
                rho.data[i] = rho.data[j]
                pre.data[i] = pre.data[j]

    def reset_mesh():
        self.tess.reset_mesh()
