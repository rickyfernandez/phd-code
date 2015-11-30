from containers.containers cimport CarrayContainer, ParticleContainer
from utils.carray cimport DoubleArray, LongLongArray

cdef class ReconstructionBase:
    def __init__(self):
        pass

    def compute(self, particles, faces, left_state, right_state, gamma, dt):
        self._compute(particles, faces, left_state, right_state, gamma, dt)

    cdef _compute(self, ParticleContainer particles, CarrayContainer faces, CarrayContainer left_state, CarrayContainer right_state,
            double gamma, double dt):
        msg = "Reconstruction::compute called!"
        raise NotImplementedError(msg)

cdef class PieceWiseConstant(ReconstructionBase):

    cdef _compute(self, ParticleContainer particles, CarrayContainer faces, CarrayContainer left_state, CarrayContainer right_state,
            double gamma, double dt):

        # particle primitive variables
        cdef DoubleArray d = particles.get_carray("density")
        cdef DoubleArray u = particles.get_carray("velocity-x")
        cdef DoubleArray v = particles.get_carray("velocity-y")
        cdef DoubleArray p = particles.get_carray("pressure")

        # left state primitive variables
        cdef DoubleArray dl = left_state.get_carray("density")
        cdef DoubleArray ul = left_state.get_carray("velocity-x")
        cdef DoubleArray vl = left_state.get_carray("velocity-y")
        cdef DoubleArray pl = left_state.get_carray("pressure")

        # left state primitive variables
        cdef DoubleArray dr = right_state.get_carray("density")
        cdef DoubleArray ur = right_state.get_carray("velocity-x")
        cdef DoubleArray vr = right_state.get_carray("velocity-y")
        cdef DoubleArray pr = right_state.get_carray("pressure")

        # particle indices that make up the face
        cdef LongLongArray pair_i = faces.get_carray("pair-i")
        cdef LongLongArray pair_j = faces.get_carray("pair-j")

        cdef int k, i, j
        cdef int num_faces = faces.get_number_of_items()

        # loop through each face
        for k in range(num_faces):

            # extract left and right particle that
            # make the face
            i = pair_i.data[k]
            j = pair_j.data[k]

            # left face values
            dl.data[k] = d.data[i]
            ul.data[k] = u.data[i]
            vl.data[k] = v.data[i]
            pl.data[k] = p.data[i]

            # right face values
            dr.data[k] = d.data[j]
            ur.data[k] = u.data[j]
            vr.data[k] = v.data[j]
            pr.data[k] = p.data[j]


#cdef class PieceWiseLinear(Reconstruction):
#    cdef _compute_gradients(self):
#
#        cdef DoubleArray x = self.particles.properties["position-x"]
#        cdef DoubleArray y = self.particles.properties["position-y"]
#        cdef DoubleArray r = self.particles.properties["density"]
#        cdef DoubleArray u = self.particles.properties["velocity-x"]
#        cdef DoubleArray v = self.particles.properties["velocity-y"]
#        cdef DoubleArray p = self.particles.properties["pressure"]
#
#        # gradients
#        cdef DoubleArray rx = self.pa.properties["x-grad-density"]
#        cdef DoubleArray ry = self.pa.properties["y-grad-density"]
#
#        cdef DoubleArray ux = self.pa.properties["x-grad-velocity-x"]
#        cdef DoubleArray uy = self.pa.properties["y-grad-velocity-x"]
#
#        cdef DoubleArray vx = self.pa.properties["x-grad-velocity-y"]
#        cdef DoubleArray vy = self.pa.properties["y-grad-velocity-y"]
#
#        cdef DoubleArray px = self.pa.properties["x-grad-pressure"]
#        cdef DoubleArray py = self.pa.properties["y-grad-pressure"]
#
#        # neighbor and face information
#        cdef int[:] neighbor_graph = self.mesh["neighbors"]
#        cdef int[:] neighbor_graph_size = self.mesh["number of neighbors"]
#        cdef int[:] face_graph = self.mesh["faces"]
#
#        cdef int[:] circum_centers = self.mesh["voronoi vertices"]
#
#        cdef int[:] center_of_mass = self.mesh[""]
#        cdef double[:] volume = self. ?
#
#        cdef int id_p, id_n, ind, ind2, ind_face, ind_face2
#        cdef int var, j
#
#        cdef double xp, yp, xn, yn, xf1, yf1, xf2, yf2
#        cdef double fx, fy, cx, cy, rx, ry
#
#        cdef double face_area, vol, r
#        cdef double dph, psi
#
#        cdef double[:] r_phi = np.zeros(2, dtype=np.float64)
#        cdef double[:] p_phi = np.zeros(2, dtype=np.float64)
#        cdef double[:] u_phi = np.zeros(2, dtype=np.float64)
#        cdef double[:] v_phi = np.zeros(2, dtype=np.float64)
#
#        cdef double r_alpha, p_alpha, u_alpha, v_alpha
#
#        ind = ind2 = 0              # neighbor index for graph
#        ind_face = ind_face2 = 0    # face index for graph
#
#        # loop over real particles
#        for i in range(self.particles.get_number_of_particles()):
#
#            # particle position 
#            _xi = x.data[i]
#            _yi = y.data[i]
#
#            # particle volume
#            _vol = vol.data[i]
#
#            # set min and max particle values
#            r_phi[0] = r_phi[1] = r.data[i]
#            p_phi[0] = p_phi[1] = p.data[i]
#            u_phi[0] = u_phi[1] = u.data[i]
#            v_phi[0] = v_phi[1] = v.data[i]
#
#            r_alpha = p_alpha = u_alpha = v_alpha = 1.0
#
#            # loop over neighbors of particle
#            for k in range(neighbor_graph_size[i]):
#
#                # index of neighbor
#                j = neighbor_graph[ind]
#
#                # neighbor position
#                _xj = x.data[j]
#                _yj = y.data[j]
#
#                # coordinates that make up the face, in 2d a
#                # face is made up of two points
#                _x1 = circum_centers[face_graph[ind_face],0]
#                _y1 = circum_centers[face_graph[ind_face],1]
#                ind_face += 1
#
#                _x2 = circum_centers[face_graph[ind_face],0]
#                _y2 = circum_centers[face_graph[ind_face],1]
#                ind_face += 1
#
#                # edge vector
#                _x = _x2 - _x1
#                _y = _y2 - _y1
#
#                face_area = sqrt(_x*_x + _y*_y)
#
#                # face center of mass
#                _fx = 0.5*(_x + _x)
#                _fy = 0.5*(_y + _y)
#
#                # face center mass relative to midpoint of particles
#                _cx = _fx - 0.5*(_xi + _xj)
#                _cy = _fy - 0.5*(_yi + _yj)
#
#                # separation vector of particles
#                _xr = xi - xn
#                _yr = yi - yn
#
#                _mag = sqrt(_xr*_xr + _yr*_yr)
#
#                # add neighbor values to max and min
#                r_phi[0] = min(r_phi[0], r.data[j])
#                p_phi[0] = min(p_phi[0], p.data[j])
#                u_phi[0] = min(u_phi[0], u.data[j])
#                v_phi[0] = min(v_phi[0], v.data[j])
#
#                r_phi[1] = max(r_phi[1], r.data[j])
#                p_phi[1] = max(p_phi[1], p.data[j])
#                u_phi[1] = max(u_phi[1], u.data[j])
#                v_phi[1] = max(v_phi[1], v.data[j])
#
#                # gradient of density
#                _rx += face_area*((r.data[j]-r.data[j])*_cx - 0.5*(r.data[i] + r.data[j])*_xr)/(_mag*_vol)
#                _ry += face_area*((r.data[j]-r.data[j])*_cy - 0.5*(r.data[i] + r.data[j])*_yr)/(_mag*_vol)
#
#                # gradient of velocities
#                _ux += face_area*((u.data[i]-u.data[j])*_cx - 0.5*(u.data[i] + u.data[j])*_xr)/(_mag*_vol)
#                _uy += face_area*((v.data[i]-v.data[j])*_cy - 0.5*(v.data[i] + v.data[j])*_yr)/(_mag*_vol)
#
#                _vx += face_area*((u.data[i]-u.data[j])*_cx - 0.5*(u.data[i] + u.data[j])*_xr)/(_mag*_vol)
#                _vy += face_area*((v.data[i]-v.data[j])*_cy - 0.5*(v.data[i] + v.data[j])*_xy)/(_mag*_vol)
#
#                # gradient of pressure
#                _px += face_area*((p.data[i]-p.data[j])*_cx - 0.5*(p.data[i] + p.data[j])*_xr)/(_mag*_vol)
#                _py += face_area*((p.data[i]-p.data[j])*_cy - 0.5*(p.data[i] + p.data[j])*_yr)/(_mag*_vol)
#
#                # go to next neighbor
#                ind += 1
#
#            for k in range(neighbor_graph_size[i]):
#
#                # index of neighbor
#                j = neighbor_graph[ind2]
#
#                _x1 = circum_centers[face_graph[ind_face2],0]
#                _y1 = circum_centers[face_graph[ind_face2],1]
#                ind_face2 += 1
#
#                _x2 = circum_centers[face_graph[ind_face2],0]
#                _y2 = circum_centers[face_graph[ind_face2],1]
#                ind_face2 += 1
#
#                _fx = 0.5*(_x1 + _x2)
#                _fy = 0.5*(_y1 + _y2)
#
#                # density
#                dphi = _rx*(_fx - cx.data[i]) + _ry*(_fy - cy.data[i])
#                if dphi > 0.0:
#                    psi = (r_phi[1] - r.data[i])/dphi
#                elif dphi < 0.0:
#                    psi = (r_phi[0] - r.data[i])/dphi
#                else:
#                    psi = 1.0
#                r_alpha = min(r_alpha, psi)
#
#                # pressure
#                dphi = _px*(_fx - cx.data[i]) + _py*(_fy - cy.data[i])
#                if dphi > 0.0:
#                    psi = (p_phi[1] - p.data[i])/dphi
#                elif dphi < 0.0:
#                    psi = (p_phi[0] - p.data[i])/dphi
#                else:
#                    psi = 1.0
#                p_alpha = min(p_alpha, psi)
#
#                # velocity
#                dphi = _ux*(_fx - cx.data[i]) + _uy*(_fy - cy.data[i])
#                if dphi > 0.0:
#                    psi = (u_phi[1] - u.data[i])/dphi
#                elif dphi < 0.0:
#                    psi = (u_phi[0] - u.data[i])/dphi
#                else:
#                    psi = 1.0
#                u_alpha = min(u_alpha, psi)
#
#                dphi = _vx*(_fx - cx.data[i]) + _vy*(_fy - cy.data[i])
#                if dphi > 0.0:
#                    psi = (v_phi[1] - v.data[i])/dphi
#                elif dphi < 0.0:
#                    psi = (v_phi[0] - v.data[i])/dphi
#                else:
#                    psi = 1.0
#                v_alpha = min(v_alpha, psi)
#
#                # go to next neighbor
#                ind2 += 1
#
#            rx.data[i] = _rx*r_alpha
#            ry.data[i] = _ry*r_alpha
#
#            px.data[i] = _px*p_alpha
#            py.data[i] = _py*p_alpha
#
#            ux.data[i] = _ux*u_alpha
#            uy.data[i] = _uy*u_alpha
#
#            vx.data[i] = _vx*v_alpha
#            vy.data[i] = _vy*v_alpha
