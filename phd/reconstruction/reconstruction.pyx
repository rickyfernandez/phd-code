import numpy as np
cimport numpy as np
cimport libc.stdlib as stdlib
from libc.math cimport sqrt, fmax, fmin

from ..mesh.mesh cimport Mesh
from ..utils.particle_tags import ParticleTAGS
from ..containers.containers cimport CarrayContainer
from ..utils.carray cimport DoubleArray, IntArray, LongLongArray, LongArray

cdef int Real = ParticleTAGS.Real

cdef class ReconstructionBase:

    def __init__(self, CarrayContainer pc, Mesh mesh):
        pass

    def compute(self, pc, faces, left_state, right_state, mesh, gamma, dt):
        self._compute(pc, faces, left_state, right_state, mesh, gamma, dt)

    cdef _compute(self, CarrayContainer pc, CarrayContainer faces, CarrayContainer left_state, CarrayContainer right_state,
            Mesh mesh, double gamma, double dt):
        msg = "Reconstruction::compute called!"
        raise NotImplementedError(msg)

cdef class PieceWiseConstant(ReconstructionBase):

    cdef _compute(self, CarrayContainer pc, CarrayContainer faces, CarrayContainer left_state, CarrayContainer right_state,
            Mesh mesh, double gamma, double dt):

        # particle primitive variables
        cdef DoubleArray d = pc.get_carray("density")
        cdef DoubleArray p = pc.get_carray("pressure")

        # left state primitive variables
        cdef DoubleArray dl = left_state.get_carray("density")
        cdef DoubleArray pl = left_state.get_carray("pressure")

        # left state primitive variables
        cdef DoubleArray dr = right_state.get_carray("density")
        cdef DoubleArray pr = right_state.get_carray("pressure")

        # particle indices that make up the face
        cdef LongArray pair_i = faces.get_carray("pair-i")
        cdef LongArray pair_j = faces.get_carray("pair-j")

        cdef int i, j, k, n
        cdef int dim = mesh.dim
        cdef np.float64_t *v[3], *vl[3], *vr[3]
        cdef int num_faces = faces.get_number_of_items()

        pc.pointer_groups(v, pc.named_groups['velocity'])
        left_state.pointer_groups(vl,  left_state.named_groups['velocity'])
        right_state.pointer_groups(vr, right_state.named_groups['velocity'])

        # loop through each face
        for n in range(num_faces):

            # extract left and right particle that
            # make the face
            i = pair_i.data[n]
            j = pair_j.data[n]

            # left face values
            dl.data[n] = d.data[i]
            pl.data[n] = p.data[i]

            # right face values
            dr.data[n] = d.data[j]
            pr.data[n] = p.data[j]

            # velocities
            for k in range(dim):
                vl[k][n] = v[k][i]
                vr[k][n] = v[k][j]

cdef class PieceWiseLinear(ReconstructionBase):

    def __init__(self, CarrayContainer pc, Mesh mesh):

        cdef int i
        cdef str field
        cdef list axis = ['x', 'y', 'z']
        cdef dict group = {}, named_groups = {}, state_vars = {}

        named_groups['primitive'] = []
        named_groups['velocity'] = []

        for field in pc.named_groups['primitive']:
            for i in range(mesh.dim):

                if field not in named_groups:
                    named_groups[field] = []

                # subset of gradient
                named_groups[field].append(field + '_' + axis[i])
                named_groups['primitive'].append(field + '_' + axis[i])

                # store velocity gradient matrix
                if 'vel' in field:
                    named_groups['velocity'].append(field + '_' + axis[i])

                state_vars[field + '_' + axis[i]] = 'double'

        self.grad = CarrayContainer(var_dict=state_vars)
        self.grad.named_groups = named_groups

    cdef _compute_gradients(self, CarrayContainer pc, CarrayContainer faces, Mesh mesh):

        # particle information
        cdef IntArray tags = pc.get_carray("tag")
        cdef DoubleArray vol = pc.get_carray("volume")

        cdef DoubleArray face_area = faces.get_carray("area")
        cdef LongArray pair_i = faces.get_carray("pair-i")
        cdef LongArray pair_j = faces.get_carray("pair-j")

        cdef double dph, psi, d_dif, d_sum
        cdef int i, j, k, n, m, fid, dim = mesh.dim

        cdef double *x[3], *dcx[3]
        cdef double cfx[3], *fij[3], area
        cdef double xi[3], xj[3], dr[3], cx[3], r, _vol

        cdef int num_fields = len(pc.named_groups['primitive'])

        cdef double[:] phi_max = np.zeros(num_fields, dtype=np.float64)
        cdef double[:] phi_min = np.zeros(num_fields, dtype=np.float64)
        cdef double[:] alpha   = np.zeros(num_fields, dtype=np.float64)
        cdef double[:,:] df = np.zeros((num_fields,dim), dtype=np.float64)

        #cdef np.float64_t *prim[4]
        #cdef np.float64_t *grad[8]

        cdef np.float64_t** prim = <np.float64_t**>stdlib.malloc(sizeof(void*)*num_fields)
        cdef np.float64_t** grad = <np.float64_t**>stdlib.malloc(sizeof(void*)*(num_fields*dim))

        # pointer to particle information
        pc.pointer_groups(x, pc.named_groups['position'])
        pc.pointer_groups(dcx, pc.named_groups['dcom'])
        pc.pointer_groups(prim, pc.named_groups['primitive'])

        # pointer to face center of mass
        faces.pointer_groups(fij, faces.named_groups['com'])

        # pointer to primitive gradients with dimension stacked
        self.grad.pointer_groups(grad, self.grad.named_groups['primitive'])

        # calculate gradients
        for i in range(pc.get_number_of_items()):
            if tags.data[i] == Real:

                # store particle position
                for k in range(dim):
                    xi[k] = x[k][i]
                    cx[k] = xi[k] + dcx[k][i]
                _vol = vol.data[i]

                for n in range(num_fields):

                    # set min/max primitive values
                    phi_max[n] = phi_min[n] = prim[n][i]
                    alpha[n]   = 1.0

                    # zero out gradients
                    for k in range(dim):
                        df[n,k] = 0

                # loop over faces of particle
                for m in range(mesh.neighbors[i].size()):

                    # index of face neighbor
                    fid = mesh.neighbors[i][m]
                    area = face_area.data[fid]

                    # extract neighbor from face
                    if i == pair_i.data[fid]:
                        j = pair_j.data[fid]
                    elif i == pair_j.data[fid]:
                        j = pair_i.data[fid]
                    else:
                        print 'error in neighobr'

                    r = 0.0
                    for k in range(dim):

                        # neighbor position
                        xj[k] = x[k][j]

                        # face center mass relative to midpoint of particles
                        cfx[k] = fij[k][fid] - 0.5*(xi[k] + xj[k])

                        # separation vector of particles
                        dr[k] = xi[k] - xj[k]
                        r += dr[k]**2

                    r = sqrt(r)

                    # extrapolate each field to face
                    for n in range(num_fields):

                        # add neighbor values to max and min
                        phi_max[n] = fmax(phi_max[n], prim[n][j])
                        phi_min[n] = fmin(phi_min[n], prim[n][j])

                        d_dif = prim[n][j] - prim[n][i]
                        d_sum = prim[n][j] + prim[n][i]

                        # gradient estimate eq. 21
                        for k in range(dim):
                            df[n,k] += area*(d_dif*cfx[k] - 0.5*d_sum*dr[k])/(r*_vol)

                # limit gradients eq. 30
                for m in range(mesh.neighbors[i].size()):

                    # index of face neighbor
                    fid = mesh.neighbors[i][m]
                    for n in range(num_fields):

                        dphi = 0
                        for k in range(dim):
                            dphi += df[n,k]*(fij[k][fid] - cx[k])

                        if dphi > 0.0:
                            psi = (phi_max[n] - prim[n][i])/dphi
                        elif dphi < 0.0:
                            psi = (phi_min[n] - prim[n][i])/dphi
                        else:
                            psi = 1.0

                        alpha[n] = fmin(alpha[n], psi)

                # store the gradients
                for n in range(num_fields):
                    for k in range(dim):
                        grad[dim*n+k][i] = alpha[n]*df[n,k]

        # transfer gradients to ghost particles
        mesh.boundary._update_gradients(pc, self.grad, self.grad.named_groups['primitive'])

        # clean up
        stdlib.free(prim)
        stdlib.free(grad)

    cdef _compute(self, CarrayContainer pc, CarrayContainer faces, CarrayContainer left_state, CarrayContainer right_state,
            Mesh mesh, double gamma, double dt):
        """
        compute linear reconstruction. Method taken from Springel (2009)
        """
        cdef double fac = 0.5*dt
        cdef double sepi, sepj, vi, vj
        cdef int i, j, k, m, n, dim = mesh.dim

        cdef np.float64_t *fij[3]
        cdef np.float64_t *vl[3], *vr[3]
        cdef np.float64_t *x[3], *v[3], *dcx[3]
        cdef np.float64_t *dd[3], *dv[9], *dp[3]

        cdef LongArray pair_i = faces.get_carray("pair-i")
        cdef LongArray pair_j = faces.get_carray("pair-j")

        # particle primitive variables
        cdef DoubleArray d = pc.get_carray("density")
        cdef DoubleArray p = pc.get_carray("pressure")

        # left state primitive variables
        cdef DoubleArray dl = left_state.get_carray("density")
        cdef DoubleArray pl = left_state.get_carray("pressure")

        # right state primitive variables
        cdef DoubleArray dr = right_state.get_carray("density")
        cdef DoubleArray pr = right_state.get_carray("pressure")

        # extract pointers
        pc.pointer_groups(x, pc.named_groups['position'])
        pc.pointer_groups(dcx, pc.named_groups['dcom'])
        pc.pointer_groups(v, pc.named_groups['velocity'])

        left_state.pointer_groups(vl,  left_state.named_groups['velocity'])
        right_state.pointer_groups(vr, left_state.named_groups['velocity'])

        faces.pointer_groups(fij, faces.named_groups['com'])

        # allocate space and compute gradients
        self.grad.resize(pc.get_number_of_items())
        self._compute_gradients(pc, faces, mesh)

        self.grad.pointer_groups(dd, self.grad.named_groups['density'])
        self.grad.pointer_groups(dv, self.grad.named_groups['velocity'])
        self.grad.pointer_groups(dp, self.grad.named_groups['pressure'])

        # create left/right states for each face
        for m in range(faces.get_number_of_items()):

            i = pair_i.data[m]
            j = pair_j.data[m]

            # density
            dl.data[m] = d.data[i]
            dr.data[m] = d.data[j]

            # pressure
            pl.data[m] = p.data[i]
            pr.data[m] = p.data[j]

            # velocity - add pressure gradient component
            for k in range(dim):
                vl[k][m] = v[k][i] - fac*dp[k][i]/d.data[i]
                vr[k][m] = v[k][j] - fac*dp[k][j]/d.data[j]

            # MUSCL schemem eq. 36
            for k in range(dim): # dot products

                # distance from particle to com of face
                sepi = fij[k][m] - (x[k][i] + dcx[k][i])
                sepj = fij[k][m] - (x[k][j] + dcx[k][j])

                vi = v[k][i]
                vj = v[k][j]

                # add gradient (eq. 21) and time extrapolation (eq. 37)
                # the trace of dv is div of velocity
                dl.data[m] += dd[k][i]*(sepi - fac*vi) - fac*d.data[i]*dv[(dim+1)*k][i]
                dr.data[m] += dd[k][j]*(sepj - fac*vj) - fac*d.data[j]*dv[(dim+1)*k][j]

                pl.data[m] += dp[k][i]*(sepi - fac*vi) - fac*gamma*p.data[i]*dv[(dim+1)*k][i]
                pr.data[m] += dp[k][j]*(sepj - fac*vj) - fac*gamma*p.data[j]*dv[(dim+1)*k][j]

                # pressure term already added before loop 
                for n in range(dim): # over velocity components
                    vl[n][m] += dv[n*dim+k][i]*(sepi - fac*vi)
                    vr[n][m] += dv[n*dim+k][j]*(sepj - fac*vj)
