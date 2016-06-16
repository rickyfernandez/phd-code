import numpy as np
cimport numpy as np

from ..utils.particle_tags import ParticleTAGS

from ..containers.containers cimport CarrayContainer, ParticleContainer
from ..utils.carray cimport DoubleArray, IntArray, LongLongArray, LongArray
from libc.math cimport sqrt, fmax, fmin
cimport libc.stdlib as stdlib
from ..mesh.mesh cimport Mesh

cdef int Real = ParticleTAGS.Real
#cdef int Boundary = ParticleTAGS.Boundary

cdef class ReconstructionBase:
    def __init__(self):
        pass

    def compute(self, particles, faces, left_state, right_state, mesh, gamma, dt):
        self._compute(particles, faces, left_state, right_state, mesh, gamma, dt)

    cdef _compute(self, ParticleContainer particles, CarrayContainer faces, CarrayContainer left_state, CarrayContainer right_state,
            Mesh mesh, double gamma, double dt):
        msg = "Reconstruction::compute called!"
        raise NotImplementedError(msg)

cdef class PieceWiseConstant(ReconstructionBase):

    cdef _compute(self, ParticleContainer particles, CarrayContainer faces, CarrayContainer left_state, CarrayContainer right_state,
            Mesh mesh, double gamma, double dt):

        # particle primitive variables
        cdef DoubleArray d = particles.get_carray("density")
        cdef DoubleArray p = particles.get_carray("pressure")

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


        particles.extract_field_vec_ptr(v, "velocity")
        left_state.extract_field_vec_ptr(vl, "velocity")
        right_state.extract_field_vec_ptr(vr, "velocity")

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


# out of comission - adding cgal library
#cdef class PieceWiseLinear(ReconstructionBase):
#
#    def __init__(self):
#
#        self.state_vars = {
#                "density": "double",
#                "velocity-x": "double",
#                "velocity-y": "double",
#                "pressure": "double",
#                }
#
#        self.gradx = CarrayContainer(var_dict=self.state_vars)
#        self.grady = CarrayContainer(var_dict=self.state_vars)
#
#    cdef _compute_gradients(self, ParticleContainer particles, CarrayContainer faces, np.int32_t[:] neighbor_graph, np.int32_t[:] num_neighbors,
#        np.int32_t[:] face_graph, double[:,::1] circum_centers):
#
#        cdef IntArray tags = particles.get_carray("tag")
#        cdef IntArray type = particles.get_carray("type")
#
#        # particle information
#        cdef DoubleArray x   = particles.get_carray("position-x")
#        cdef DoubleArray y   = particles.get_carray("position-y")
#        cdef DoubleArray cx  = particles.get_carray("com-x")
#        cdef DoubleArray cy  = particles.get_carray("com-y")
#        cdef DoubleArray vol = particles.get_carray("volume")
#
#        cdef DoubleArray pres = particles.get_carray("pressure")
#
#        cdef double _xi, _yi, _xj, _yj, _xr, _yr, _cx, _cy, _mag, _vol
#        cdef double _x1, _y1, _x2, _y2, _fpx, _fpy, _fcx, _fcy, face_area
#        cdef double dph, psi
#        cdef int i, j, k, ind, ind2, ind_face, ind_face2
#        cdef int num_fields
#
#        cdef double[:] phi_max = np.zeros(4, dtype=np.float64)
#        cdef double[:] phi_min = np.zeros(4, dtype=np.float64)
#        cdef double[:] alpha   = np.zeros(4, dtype=np.float64)
#
#        ind = ind2 = 0              # neighbor index for graph
#        ind_face = ind_face2 = 0    # face index for graph
#
#        num_fields = len(self.state_vars)
#
#        # allocate pointers to fields for indexing
#        cdef np.float64_t** fields = <np.float64_t**>stdlib.malloc(sizeof(void*)*num_fields)
#        cdef np.float64_t** dfdx   = <np.float64_t**>stdlib.malloc(sizeof(void*)*num_fields)
#        cdef np.float64_t** dfdy   = <np.float64_t**>stdlib.malloc(sizeof(void*)*num_fields)
#
#        cdef str name
#        cdef DoubleArray tmp
#        cdef int var
#        for i, name in enumerate(self.state_vars):
#
#            # store field pointers to array
#            tmp = particles.get_carray(name);  fields[i] = tmp.get_data_ptr()
#            tmp = self.gradx.get_carray(name); dfdx[i]   = tmp.get_data_ptr()
#            tmp = self.grady.get_carray(name); dfdy[i]   = tmp.get_data_ptr()
#
#        # loop over particles
#        for i in range(particles.get_number_of_particles()):
#            if tags.data[i] == Real or type.data[i] == Boundary:
#
#                # particle position 
#                _xi = x.data[i]
#                _yi = y.data[i]
#
#                # particle volume
#                _vol = vol.data[i]
#                if vol.data[i] == 0:
#                    print 'tag:', tags.data[i], 'type:', type.data[i]
#                    raise RuntimeError("volume equal to zero")
#
#                for var in range(num_fields):
#
#                    # set min and max particle values
#                    phi_max[var] = fields[var][i]
#                    phi_min[var] = fields[var][i]
#                    alpha[var]   = 1.0
#
#                    # zero out gradients
#                    dfdx[var][i] = 0
#                    dfdy[var][i] = 0
#
#                # loop over neighbors of particle
#                for k in range(num_neighbors[i]):
#
#                    # index of neighbor
#                    j = neighbor_graph[ind]
#
#                    # neighbor position
#                    _xj = x.data[j]
#                    _yj = y.data[j]
#
#                    # coordinates that make up the face, in 2d a
#                    # face is made up of two points
#                    _x1 = circum_centers[face_graph[ind_face],0]
#                    _y1 = circum_centers[face_graph[ind_face],1]
#                    ind_face += 1 # go to next face vertex
#
#                    _x2 = circum_centers[face_graph[ind_face],0]
#                    _y2 = circum_centers[face_graph[ind_face],1]
#                    ind_face += 1 # go to next face
#
#                    face_area = sqrt( (_x2 - _x1)*(_x2 - _x1) +\
#                            (_y2 - _y1)*(_y2 - _y1) )
#
#                    # face center of mass
#                    _fcx = 0.5*(_x1 + _x2)
#                    _fcy = 0.5*(_y1 + _y2)
#
#                    # face center mass relative to midpoint of particles
#                    _fpx = _fcx - 0.5*(_xi + _xj)
#                    _fpy = _fcy - 0.5*(_yi + _yj)
#
#                    # separation vector of particles
#                    _xr = _xi - _xj
#                    _yr = _yi - _yj
#                    _mag = sqrt(_xr*_xr + _yr*_yr)
#
#                    if _mag == 0:
#                        print 'tag:', tags.data[i], 'type:', type.data[i], 'i:', i, 'j:', j
#                        raise RuntimeError("mag equal to zero")
#
#                    for var in range(num_fields):
#
#                        # add neighbor values to max and min
#                        phi_max[var] = fmax(phi_max[var], fields[var][j])
#                        phi_min[var] = fmin(phi_min[var], fields[var][j])
#
#                        dfdx[var][i] += face_area*( (fields[var][j] -\
#                                fields[var][i])*_fpx - 0.5*(fields[var][i] + fields[var][j])*_xr )/(_mag*_vol)
#                        dfdy[var][i] += face_area*( (fields[var][j] -\
#                                fields[var][i])*_fpy - 0.5*(fields[var][i] + fields[var][j])*_yr )/(_mag*_vol)
#
#                    # go to next neighbor
#                    ind += 1
#
#                # center of mass of particle
#                _cx = cx.data[i]
#                _cy = cy.data[i]
#
#                for k in range(num_neighbors[i]):
#
#                    # index of neighbor
#                    j = neighbor_graph[ind2]
#
#                    _x1 = circum_centers[face_graph[ind_face2],0]
#                    _y1 = circum_centers[face_graph[ind_face2],1]
#                    ind_face2 += 1
#
#                    _x2 = circum_centers[face_graph[ind_face2],0]
#                    _y2 = circum_centers[face_graph[ind_face2],1]
#                    ind_face2 += 1
#
#                    # face center of mass
#                    _fcx = 0.5*(_x1 + _x2)
#                    _fcy = 0.5*(_y1 + _y2)
#
#                    for var in range(num_fields):
#
#                        dphi = dfdx[var][i]*(_fcx - _cx) + dfdy[var][i]*(_fcy - _cy)
#                        if dphi > 0.0:
#                            psi = (phi_max[var] - fields[var][i])/dphi
#                        elif dphi < 0.0:
#                            psi = (phi_min[var] - fields[var][i])/dphi
#                        else:
#                            psi = 1.0
#
#                        alpha[var] = fmin(alpha[var], psi)
#
#                    # go to next neighbor
#                    ind2 += 1
#
#                for var in range(num_fields):
#
#                    dfdx[var][i] *= alpha[var]
#                    dfdy[var][i] *= alpha[var]
#            else:
#
#                ind  += num_neighbors[i]
#                ind2 += num_neighbors[i]
#                ind_face  += num_neighbors[i]*2
#                ind_face2 += num_neighbors[i]*2
#
#        # relase pointers
#        stdlib.free(fields)
#        stdlib.free(dfdx)
#        stdlib.free(dfdy)
#
#    cdef _compute(self, ParticleContainer particles, CarrayContainer faces, CarrayContainer left_state, CarrayContainer right_state,
#            object mesh, double gamma, double dt):
#
#        cdef int i, j, k, var
#        cdef double _fcx, _fcy, _cxi, _cxj, _cyi, _cyj
#
#        # particle indices that make up the face
#        cdef LongLongArray pair_i = faces.get_carray("pair-i")
#        cdef LongLongArray pair_j = faces.get_carray("pair-j")
#        cdef DoubleArray fcx  = faces.get_carray("com-x")
#        cdef DoubleArray fcy  = faces.get_carray("com-y")
#
#        # particle primitive variables
#        cdef DoubleArray r = particles.get_carray("density")
#        cdef DoubleArray u = particles.get_carray("velocity-x")
#        cdef DoubleArray v = particles.get_carray("velocity-y")
#        cdef DoubleArray p = particles.get_carray("pressure")
#
#        cdef DoubleArray cx  = particles.get_carray("com-x")
#        cdef DoubleArray cy  = particles.get_carray("com-y")
#
#        # left state primitive variables
#        cdef DoubleArray rl = left_state.get_carray("density")
#        cdef DoubleArray ul = left_state.get_carray("velocity-x")
#        cdef DoubleArray vl = left_state.get_carray("velocity-y")
#        cdef DoubleArray pl = left_state.get_carray("pressure")
#
#        # left state primitive variables
#        cdef DoubleArray rr = right_state.get_carray("density")
#        cdef DoubleArray ur = right_state.get_carray("velocity-x")
#        cdef DoubleArray vr = right_state.get_carray("velocity-y")
#        cdef DoubleArray pr = right_state.get_carray("pressure")
#
#        cdef DoubleArray drdx = self.gradx.get_carray("density")
#        cdef DoubleArray dudx = self.gradx.get_carray("velocity-x")
#        cdef DoubleArray dvdx = self.gradx.get_carray("velocity-y")
#        cdef DoubleArray dpdx = self.gradx.get_carray("pressure")
#
#        cdef DoubleArray drdy = self.grady.get_carray("density")
#        cdef DoubleArray dudy = self.grady.get_carray("velocity-x")
#        cdef DoubleArray dvdy = self.grady.get_carray("velocity-y")
#        cdef DoubleArray dpdy = self.grady.get_carray("pressure")
#
#        cdef int num_particles = particles.get_number_of_particles()
#
#        self.gradx.resize(num_particles)
#        self.grady.resize(num_particles)
#
#        self._compute_gradients(particles, faces, mesh['neighbors'], mesh['number of neighbors'],
#                mesh['faces'], mesh['voronoi vertices'])
#
#        for k in range(faces.get_number_of_items()):
#
#            i = pair_i.data[k]
#            j = pair_j.data[k]
#
#            # density
#            rl.data[k] = r.data[i] - 0.5*dt*( u.data[i]*drdx.data[i] + v.data[i]*drdy.data[i] + r.data[i]*(dudx[i] + dvdy[i]) )
#            rr.data[k] = r.data[j] - 0.5*dt*( u.data[j]*drdx.data[j] + v.data[j]*drdy.data[j] + r.data[j]*(dudx[j] + dvdy[j]) )
#
#            # velocity x
#            ul.data[k] = u.data[i] - 0.5*dt*( u.data[i]*dudx.data[i] + v.data[i]*dudy.data[i] + dpdx.data[i]/r.data[i] )
#            ur.data[k] = u.data[j] - 0.5*dt*( u.data[j]*dudx.data[j] + v.data[j]*dudy.data[j] + dpdx.data[j]/r.data[j] )
#
#            # velocity y
#            vl.data[k] = v.data[i] - 0.5*dt*( u.data[i]*dvdx.data[i] + v.data[i]*dvdy.data[i] + dpdy.data[i]/r.data[i] )
#            vr.data[k] = v.data[j] - 0.5*dt*( u.data[j]*dvdx.data[j] + v.data[j]*dvdy.data[j] + dpdy.data[j]/r.data[j] )
#
#            # pressure
#            pl.data[k] = p.data[i] - 0.5*dt*( u.data[i]*dpdx.data[i] + v.data[i]*dpdy.data[i] + gamma*p.data[i]*(dudx.data[i] + dvdy.data[i]) )
#            pr.data[k] = p.data[j] - 0.5*dt*( u.data[j]*dpdx.data[j] + v.data[j]*dpdy.data[j] + gamma*p.data[j]*(dudx.data[j] + dvdy.data[j]) )
#
#            # spatial component
###            for var in range(4):
###
#            _fcx = fcx.data[k]; _fcy = fcy.data[k]
#            _cxi =  cx.data[i]; _cxj =  cx.data[j]
#            _cyi =  cy.data[i]; _cyj =  cy.data[j]
#
#            rl.data[k] += drdx.data[i]*(_fcx - _cxi) + drdy.data[i]*(_fcy - _cyi)
#            rr.data[k] += drdx.data[j]*(_fcx - _cxj) + drdy.data[j]*(_fcy - _cyj)
#
#            ul.data[k] += dudx.data[i]*(_fcx - _cxi) + dudy.data[i]*(_fcy - _cyi)
#            ur.data[k] += dudx.data[j]*(_fcx - _cxj) + dudy.data[j]*(_fcy - _cyj)
#
#            vl.data[k] += dvdx.data[i]*(_fcx - _cxi) + dvdy.data[i]*(_fcy - _cyi)
#            vr.data[k] += dvdx.data[j]*(_fcx - _cxj) + dvdy.data[j]*(_fcy - _cyj)
#
#            pl.data[k] += dpdx.data[i]*(_fcx - _cxi) + dpdy.data[i]*(_fcy - _cyi)
#            pr.data[k] += dpdx.data[j]*(_fcx - _cxj) + dpdy.data[j]*(_fcy - _cyj)
#
###                left_face[var,k]  += gradx[var,i]*(face_com[0,k] - cell_com[0,i]) + grady[var,i]*(face_com[1,k] - cell_com[1,i])
###                right_face[var,k] += gradx[var,j]*(face_com[0,k] - cell_com[0,j]) + grady[var,j]*(face_com[1,k] - cell_com[1,j])
###
###                if (left_face[var,k] < 0.0) and (var == 0 or var == 3):
###                    print "left_face[", var, "],", k, "] = ", left_face[var,k]
###
###                if (right_face[var,k] < 0.0) and (var == 0 or var == 3):
###                    print "right_face[", var, "],", k, "] = ", right_face[var,k]
