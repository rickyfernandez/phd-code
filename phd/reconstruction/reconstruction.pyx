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

        cdef long k, i, j
        cdef long num_faces = faces.get_number_of_items()

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
#        cdef DoubleArray x = self.pa.properties["position-x"]
#        cdef DoubleArray y = self.pa.properties["position-y"]
#        cdef DoubleArray r = self.pa.properties["density"]
#        cdef DoubleArray u = self.pa.properties["velocity-x"]
#        cdef DoubleArray v = self.pa.properties["velocity-y"]
#        cdef DoubleArray p = self.pa.properties["pressure"]
#
#        # gradients
#        cdef DoubleArray rx = self.pa.properties["x-grad_density"]
#        cdef DoubleArray ry = self.pa.properties["y-grad_density"]
#
#        cdef DoubleArray ux = self.pa.properties["x-grad_velocity-x"]
#        cdef DoubleArray uy = self.pa.properties["y-grad_velocity-x"]
#
#        cdef DoubleArray vx = self.pa.properties["x-grad_velocity-y"]
#        cdef DoubleArray vy = self.pa.properties["y-grad_velocity-y"]
#
#        cdef DoubleArray px = self.pa.properties["x-grad_pressure"]
#        cdef DoubleArray py = self.pa.properties["y-grad_pressure"]
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
#        for id_p in range(num_real_particles):
#
#            # particle position 
#            xp = x.data[id_p]
#            yp = y.data[id_p]
#
#            # particle volume
#            vol = volume[id_p]
#
#            # set min particle values
#            r_phi[0] = r.data[id_p]
#            p_phi[0] = p.data[id_p]
#            u_phi[0] = u.data[id_p]
#            v_phi[0] = v.data[id_p]
#
#            # set max particle values
#            r_phi[1] = r.data[id_p]
#            p_phi[1] = p.data[id_p]
#            u_phi[1] = u.data[id_p]
#            v_phi[1] = v.data[id_p]
#
#            r_alpha = p_alpha = u_alpha = v_alpha = 1.0
#
#            # loop over neighbors of particle
#            for j in range(neighbor_graph_size[id_p]):
#
#                # index of neighbor
#                id_n = neighbor_graph[ind]
#
#                # neighbor position
#                xn = x.data[id_n]
#                yn = y.data[id_n]
#
#                # coordinates that make up the face, in 2d a
#                # face is made up of two points
#                xf1 = circum_centers[face_graph[ind_face],0]
#                yf1 = circum_centers[face_graph[ind_face],1]
#                ind_face += 1
#
#                xf2 = circum_centers[face_graph[ind_face],0]
#                yf2 = circum_centers[face_graph[ind_face],1]
#                ind_face += 1
#
#                face_area = sqrt((xf2-xf1)*(xf2-xf1) + (yf2-yf1)*(yf2-yf1))
#
#                # face center of mass
#                fx = 0.5*(xf1 + xf2)
#                fy = 0.5*(yf1 + yf2)
#
#                # face center mass relative to midpoint of particles
#                cx = fx - 0.5*(xp + xn)
#                cy = fy - 0.5*(yp + yn)
#
#                # separation vector of particles
#                rvec_x = xp - xn
#                rvec_y = yp - yn
#                rvec_m = sqrt(rx*rx + ry*ry)
#
#                # add neighbor values to max and min
#                r_phi[0] = min(r_phi[0], r.data[id_n])
#                p_phi[0] = min(p_phi[0], p.data[id_n])
#                u_phi[0] = min(u_phi[0], u.data[id_n])
#                v_phi[0] = min(v_phi[0], v.data[id_n])
#
#                r_phi[1] = min(r_phi[1], r.data[id_n])
#                p_phi[1] = min(p_phi[1], p.data[id_n])
#                u_phi[1] = min(u_phi[1], u.data[id_n])
#                v_phi[1] = min(v_phi[1], v.data[id_n])
#
#                # gradient of density
#                _rx += face_area*((r.data[id_n]-r.data[id_p])*cx - 0.5*(r.data[id_p] + r.data[id_n])*rvec_x)/(rvec_m*vol)
#                _ry += face_area*((r.data[id_n]-r.data[id_p])*cy - 0.5*(r.data[id_p] + r.data[id_n])*rvec_y)/(rvec_m*vol)
#
#                # gradient of density
#                _px += face_area*((p.data[id_n]-p.data[id_p])*cx - 0.5*(p.data[id_p] + p.data[id_n])*rvec_x)/(rvec_m*vol)
#                _py += face_area*((p.data[id_n]-p.data[id_p])*cy - 0.5*(p.data[id_p] + p.data[id_n])*rvec_y)/(rvec_m*vol)
#
#                # gradient of velocities
#                _ux += face_area*((u.data[id_n]-u.data[id_p])*cx - 0.5*(u.data[id_p] + u.data[id_n])*rvec_x)/(rvec_m*vol)
#                _uy += face_area*((v.data[id_n]-v.data[id_p])*cy - 0.5*(v.data[id_p] + v.data[id_n])*rvec_y)/(rvec_m*vol)
#
#                _vx += face_area*((u.data[id_n]-u.data[id_p])*cx - 0.5*(u.data[id_p] + u.data[id_n])*rvec_x)/(rvec_m*vol)
#                _vy += face_area*((v.data[id_n]-v.data[id_p])*cy - 0.5*(v.data[id_p] + v.data[id_n])*rvec_y)/(rvec_m*vol)
#
#                # go to next neighbor
#                ind += 1
#
#
#            for j in range(neighbor_graph_size[id_p]):
#
#                # index of neighbor
#                id_n = neighbor_graph[ind2]
#
#                xf1 = circum_centers[face_graph[ind_face2],0]
#                yf1 = circum_centers[face_graph[ind_face2],1]
#                ind_face2 += 1
#
#                xf2 = circum_centers[face_graph[ind_face2],0]
#                yf2 = circum_centers[face_graph[ind_face2],1]
#                ind_face2 += 1
#
#                fx = 0.5*(xf1 + xf2)
#                fy = 0.5*(yf1 + yf2)
#
#                # density
#                dphi = _rx*(fx - center_of_mass[0,id_p]) + _ry*(fy - center_of_mass[1,id_p])
#                if dphi > 0.0:
#                    psi = (r_phi[1] - r.data[id_p])/dphi
#                elif dphi < 0.0:
#                    psi = (r_phi[0] - r.data[id_p])/dphi
#                else:
#                    psi = 1.0
#                r_alpha = min(r_alpha, psi)
#
#                # pressure
#                dphi = _px*(fx - center_of_mass[0,id_p]) + _py*(fy - center_of_mass[1,id_p])
#                if dphi > 0.0:
#                    psi = (p_phi[1] - p.data[id_p])/dphi
#                elif dphi < 0.0:
#                    psi = (p_phi[0] - p.data[id_p])/dphi
#                else:
#                    psi = 1.0
#                p_alpha = min(p_alpha, psi)
#
#                # velocity
#                dphi = _ux*(fx - center_of_mass[0,id_p]) + _uy*(fy - center_of_mass[1,id_p])
#                if dphi > 0.0:
#                    psi = (u_phi[1] - u.data[id_p])/dphi
#                elif dphi < 0.0:
#                    psi = (u_phi[0] - u.data[id_p])/dphi
#                else:
#                    psi = 1.0
#                u_alpha = min(u_alpha, psi)
#
#                dphi = _vx*(fx - center_of_mass[0,id_p]) + _vy*(fy - center_of_mass[1,id_p])
#                if dphi > 0.0:
#                    psi = (v_phi[1] - v.data[id_p])/dphi
#                elif dphi < 0.0:
#                    psi = (v_phi[0] - v.data[id_p])/dphi
#                else:
#                    psi = 1.0
#                v_alpha = min(v_alpha, psi)
#
#                # go to next neighbor
#                ind2 += 1
#
#            rx.data[id_p] = _rx*r_alpha
#            ry.data[id_p] = _ry*r_alpha
#
#            px.data[id_p] = _px*p_alpha
#            py.data[id_p] = _py*p_alpha
#
#            ux.data[id_p] = _ux*u_alpha
#            uy.data[id_p] = _uy*u_alpha
#
#            vx.data[id_p] = _vx*v_alpha
#            vy.data[id_p] = _vy*v_alpha
