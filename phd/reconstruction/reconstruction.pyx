import numpy as np
from collections import defaultdict

cimport numpy as np
cimport libc.stdlib as stdlib
from libc.math cimport sqrt, fmax, fmin

from ..utils.particle_tags import ParticleTAGS
from ..utils.carray cimport DoubleArray, IntArray, LongArray

cdef class ReconstructionBase:
    def __init__(self):
        self.registered_fields = False

    def initialize(self):
        """
        Setup all connections for computation classes. Should check always
        check if registered_fields is True.
        """
        msg = "Reconstruction::initialize called!"
        raise NotImplementedError(msg)

    def set_fields_for_reconstruction(self, CarrayContainer particles):
        """
        Create lists of variables to reconstruct and setup containers for
        gradients and reconstructions
        """
        cdef str field, dtype
        cdef dict field_types = {}, named_groups = defaultdict([])

        # add standard primitive fields
        for field in particles.named_groups["primitive"]:

            # add field to group
            named_groups["primitive"].append(field)

            # check type of field
            dtype = particles.carray_info[field]
            if dtype != "double":
                raise RuntimeError(
                        "Reconstruction: %field non double type" % dtype)
            field_types[field] = "double"

        named_groups["velocity"] = particles.named_groups["velocity"]

        # store fields info
        self.registered_fields = True
        self.reconstruct_fields = field_types
        self.reconstruct_field_groups = named_groups

    cpdef compute_states(self, CarrayContainer particles, Mesh mesh, EquationStateBase eos,
            RiemannBase riemann, DomainManager domain_manager, double dt, int dim):
        """
        Perform reconstruction from cell center to face center of each face in
        mesh.
        """
        msg = "Reconstruction::compute called!"
        raise NotImplementedError(msg)

cdef class PieceWiseConstant(ReconstructionBase):
    def __init__(self):
        super(PieceWiseConstant, self).__init__()

    def initialize(self):
        if not self.registered_fields:
            raise RuntimeError(
                    "Reconstruction did not set fields to reconstruct!")

        # initialize left/right face states for riemann solver
        self.left_states  = CarrayContainer(var_dict=self.reconstruct_fields)
        self.right_states = CarrayContainer(var_dict=self.reconstruct_fields)

        # add groups
        self.left_states.named_groups  = self.reconstruct_field_groups
        self.right_states.named_groups = self.reconstruct_field_groups

    cpdef compute_states(self, CarrayContainer particles, Mesh mesh, EquationStateBase eos,
            RiemannBase riemann, DomainManager domain_manager, double dt, int dim):
        """Construct left and right states for riemann solver for each face"""

        # particle primitive variables
        cdef DoubleArray d = particles.get_carray("density")
        cdef DoubleArray p = particles.get_carray("pressure")

        # left state primitive variables
        cdef DoubleArray dl = self.left_states.get_carray("density")
        cdef DoubleArray pl = self.left_states.get_carray("pressure")

        # right state primitive variables
        cdef DoubleArray dr = self.right_states.get_carray("density")
        cdef DoubleArray pr = self.right_states.get_carray("pressure")

        # particle indices that make up face
        cdef LongArray pair_i = mesh.faces.get_carray("pair-i")
        cdef LongArray pair_j = mesh.faces.get_carray("pair-j")

        cdef int i, j, k, n
        cdef bint boost = riemann.param_boost
        cdef np.float64_t *v[3], *vl[3], *vr[3], *wx[3]

        # particle and face velocity pointer
        particles.pointer_groups(v, particles.named_groups["velocity"])
        mesh.faces.pointer_groups(wx, mesh.faces.named_groups["velocity"])

        # face state velocity pointer
        self.left_states.pointer_groups(vl,  self.reconstruct_field_groups["velocity"])
        self.right_states.pointer_groups(vr, self.reconstruct_field_groups["velocity"])

        # loop through each face
        for n in range(mesh.faces.get_number_of_items()):

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
                if boost:
                    vl[k][n] = v[k][i] - wx[k][n]
                    vr[k][n] = v[k][j] - wx[k][n]
                else:
                    vl[k][n] = v[k][i]
                    vr[k][n] = v[k][j]

#cdef class PieceWiseLinear(ReconstructionBase):
#    def __init__(self, int param_limiter = 0):
#        self.param_limiter = param_limiter
#
#    def __dealloc__(self):
#        """Release pointers"""
#
#        stdlib.free(self.prim_ptr)
#        stdlib.free(self.grad_ptr)
#
#        if self.do_colors:
#            stdlib.free(self.col)
#            stdlib.free(self.coll)
#            stdlib.free(self.colr)
#            stdlib.free(self.dc)
#
#        stdlib.free(self.phi_max)
#        stdlib.free(self.phi_min)
#
#        stdlib.free(self.alpha)
#        stdlib.free(self.df)
#
#    def set_fields_to_reconstruct(CarrayContainer particles):
#        """
#        Create lists of variables to reconstruct and setup containers for
#        gradients and reconstructions
#        """
#        cdef str field, grad_name
#        cdef int i, num_fields, dim
#        cdef list axis = ["x", "y", "z"]
#        cdef dict grad_groups = defaultdict([])
#        cdef dict state_vars = {}, grad_vars = {}
#
#        dim = particles.info["dim"]
#
#        # add primitive fields
#        for field in particles.named_groups["primitive"]:
#            state_vars[field] = "double"
#            for i in range(dim):
#
#                # store gradient of field
#                grad_name = field + "_" + axis[i]               # field name
#                grad_vars[grad_name] = "double"                 # creation list
#                grad_groups[field].append(grad_name)            # gradient vector list
#                grad_groups["primitive"].append(grad_name)      # main list
#
#                # store velocity gradient matrix
#                if "vel" in field:
#                    grad_groups["velocity"].append(grad_name)
#
#        # add species fields
#        if "species" in particles.named_groups.keys():
#            for field in particles.named_groups["species"]:
#                state_vars[field] = "double"
#                for i in range(dim):
#
#                    # store gradient of field
#                    grad_name = field + "_" + axis[i]           # field name
#                    grad_vars[grad_name] = "double"             # creation list
#                    grad_groups[field].append(grad_name)        # gradient vector list
#                    grad_groups["primitive"].append(grad_name)  # main list
#                    grad_groups["species"].append(gradd_name)
#
#        # add species fields
#        if "passive-scalars" in particles.named_groups.keys():
#            for field in particles.named_groups["passive-scalars"]:
#                state_vars[field] = "double"
#                for i in range(dim):
#
#                    # store gradient of field
#                    grad_name = field + "_" + axis[i]           # field name
#                    state_vars[grad_name] = "double"            # creation list
#                    grad_groups[field].append(grad_name)        # gradient vector list
#                    grad_groups["primitive"].append(grad_name)  # main list
#                    grad_groups["passive-scalars"].append(grad_name)
#
#        # store fields
#        self.reconstruct_fields = state_vars
#        self.reconstruct_grad_groups = grad_vars
#        self.reconstruct_field_groups = named_groups
#
#    def initialize(self):
#        """Setup initial arrays and routines for computation"""
#
#        if self.fields_to_reconstruct or\
#                self.fields_to_reconstruct_grad or\
#                self.fields_to_reconstruct_groups:
#            raise RuntimeError("fields to reconstruct not specified")
#
#        # create states
#        self.left_state  = CarrayContainer(var_dict=self.fields_to_reconstruct)
#        self.right_state = CarrayContainer(var_dict=self.fields_to_reconstruct)
#        self.grad = CarrayContainer(var_dict=self.grad_state_vars)
#
#        # allocate helper pointers
#        dim = len(named_groups["velocity"])
#        num_fields = len(self.fields_to_reconstruct_groups["primitive"])
#
#        if self.do_colors:
#            num_colors = len(self.fields_to_reconstruct_groups["colors"])
#            self.cl = <np.float64_t**> stdlib.malloc(num_colors*sizeof(void*))
#            self.cr = <np.float64_t**> stdlib.malloc(num_colors*sizeof(void*))
#            self.dc = <np.flota64_t**> stdlib.malloc((num_colors*dim)*sizeof(void*))
#
#        # primitive values and gradient
#        self.prim_ptr = <np.float64_t**> stdlib.malloc(num_fields*sizeof(void*))
#        self.grad_ptr = <np.float64_t**> stdlib.malloc((num_fields*dim)*sizeof(void*))
#
#        # min/max of field value of particle
#        self.phi_max = <np.float64_t*> stdlib.malloc(num_fields*sizeof(np.float64))
#        self.phi_min = <np.float64_t*> stdlib.malloc(num_fields*sizeof(np.float64))
#        self.alpha   = <np.float64_t*> stdlib.malloc(num_fields*sizeof(np.float64))
#
#        # difference of field value at paticle position to face position
#        self.df = <np.float64_t*> stdlib.malloc((num_fields*dim)*sizeof(np.float64))
#
#    cdef compute_gradients(self, CarrayContainer particles, Mesh mesh):
#        """Compute gradients for each primitive variable"""
#
#        # particle information
#        cdef IntArray flags = particles.get_carray("flag")
#        cdef DoubleArray vol = particles.get_carray("volume")
#
#        cdef DoubleArray face_area = mesh.faces.get_carray("area")
#        cdef LongArray pair_i = mesh.faces.get_carray("pair-i")
#        cdef LongArray pair_j = mesh.faces.get_carray("pair-j")
#
#        cdef int limiter = self.param_limiter
#
#        cdef double dph, psi, d_dif, d_sum
#        cdef int i, j, k, n, m, fid, dim = particles.info["dim"]
#
#        cdef double *x[3], *dcx[3]
#        cdef double cfx[3], *fij[3], area
#        cdef double xi[3], xj[3], dr[3], cx[3], r, _vol
#
#        cdef int num_fields = len(particles.named_groups['primitive'])
#
#        cdef np.float64_t** prim = self.prim_ptr
#        cdef np.float64_t** grad = self.grad_ptr
#
#        cdef np.float64_t* phi_max = self.phi_max
#        cdef np.float64_t* phi_min = self.phi_min
#        cdef np.float64_t* alpha   = self.alpha
#        cdef np.float64_t* df      = self.df
#
#        # pointer to particle information
#        particles.pointer_groups(x, particles.named_groups['position'])
#        particles.pointer_groups(dcx, particles.named_groups['dcom'])
#        particles.pointer_groups(prim, particles.named_groups['primitive'])
#
#        # pointer to face center of mass
#        mesh.faces.pointer_groups(fij, mesh.faces.named_groups['com'])
#
#        # pointer to primitive gradients with dimension stacked
#        self.grad.pointer_groups(grad, self.grad.named_groups['primitive'])
#
#        # calculate gradients
#        for i in range(particles.get_number_of_items()):
#            if(flags.data[i] & Real):
#
#                # store particle position
#                for k in range(dim):
#                    xi[k] = x[k][i]
#                    cx[k] = xi[k] + dcx[k][i]
#                _vol = vol.data[i]
#
#                for n in range(num_fields):
#
#                    # set min/max primitive values
#                    phi_max[n] = phi_min[n] = prim[n][i]
#                    alpha[n]   = 1.0
#
#                    # zero out gradients
#                    for k in range(dim):
#                        df[dim*n+k] = 0
#
#                # loop over faces of particle
#                for m in range(mesh.neighbors[i].size()):
#
#                    # index of face neighbor
#                    fid = mesh.neighbors[i][m]
#                    area = face_area.data[fid]
#
#                    # extract neighbor from face
#                    if i == pair_i.data[fid]:
#                        j = pair_j.data[fid]
#                    elif i == pair_j.data[fid]:
#                        j = pair_i.data[fid]
#                    else:
#                        print 'error in neighobr'
#
#                    r = 0.0
#                    for k in range(dim):
#
#                        # neighbor position
#                        xj[k] = x[k][j]
#
#                        # face center mass relative to midpoint of particles
#                        cfx[k] = fij[k][fid] - 0.5*(xi[k] + xj[k])
#
#                        # separation vector of particles
#                        dr[k] = xi[k] - xj[k]
#                        r += dr[k]**2
#
#                    r = sqrt(r)
#
#                    # extrapolate each field to face
#                    for n in range(num_fields):
#
#                        # add neighbor values to max and min
#                        phi_max[n] = fmax(phi_max[n], prim[n][j])
#                        phi_min[n] = fmin(phi_min[n], prim[n][j])
#
#                        d_dif = prim[n][j] - prim[n][i]
#                        d_sum = prim[n][j] + prim[n][i]
#
#                        # gradient estimate eq. 21
#                        for k in range(dim):
#                            df[dim*n+k] += area*(d_dif*cfx[k] - 0.5*d_sum*dr[k])/(r*_vol)
#
#                # limit gradients eq. 30
#                for m in range(mesh.neighbors[i].size()):
#
#                    # index of face neighbor
#                    fid = mesh.neighbors[i][m]
#
#                    if limiter == 0: # AREPO limiter
#
#                        for n in range(num_fields):
#
#                            dphi = 0
#                            for k in range(dim):
#                                dphi += df[dim*n+k]*(fij[k][fid] - cx[k])
#
#                            if dphi > 0.0:
#                                psi = (phi_max[n] - prim[n][i])/dphi
#                            elif dphi < 0.0:
#                                psi = (phi_min[n] - prim[n][i])/dphi
#                            else:
#                                psi = 1.0
#
#                            alpha[n] = fmin(alpha[n], psi)
#
#                    elif limiter == 1: # TESS limiter
#
#                        for n in range(num_fields):
#
#                            # extract neighbor from face
#                            if i == pair_i.data[fid]:
#                                j = pair_j.data[fid]
#                            elif i == pair_j.data[fid]:
#                                j = pair_i.data[fid]
#
#                            dphi = 0
#                            for k in range(dim):
#                                dphi += df[dim*n+k]*(fij[k][fid] - cx[k])
#
#                            if dphi > 0.0:
#                                psi = max((prim[n][j] - prim[n][i])/dphi, 0.)
#                            elif dphi < 0.0:
#                                psi = max((prim[n][j] - prim[n][i])/dphi, 0.)
#                            else:
#                                psi = 1.0
#
#                            alpha[n] = fmin(alpha[n], psi)
#
#                # store the gradients
#                for n in range(num_fields):
#                    for k in range(dim):
#                        grad[dim*n+k][i] = alpha[n]*df[dim*n+k]
#
#        # transfer gradients to ghost particles
#        domain_manager.update_gradients(particles, self.grad, self.grad.named_groups['primitive'])
#
#    def compute_states(self, CarrayContainer particles, Mesh mesh, EquationStateBase eos,
#            RiemannBase riemann, double dt):
#        """
#        compute linear reconstruction. Method taken from Springel (2009)
#        """
#        cdef double fac = 0.5*dt
#        cdef bint boost = riemann.param_boost
#        cdef double sepi, sepj, gamma = eos.get_gamma()
#        cdef int i, j, k, m, n, dim = particles.info["dim"]
#
#        cdef np.float64_t vi[3], vj[3]
#        cdef np.float64_t *fij[3], *wx[3]
#        cdef np.float64_t *vl[3], *vr[3]
#        cdef np.float64_t *x[3], *v[3], *dcx[3]
#        cdef np.float64_t *dd[3], *dv[9], *dp[3]
#
#        cdef LongArray pair_i = mesh.faces.get_carray("pair-i")
#        cdef LongArray pair_j = mesh.faces.get_carray("pair-j")
#
#        # particle primitive variables
#        cdef DoubleArray d = particles.get_carray("density")
#        cdef DoubleArray p = particles.get_carray("pressure")
#
#        # left state primitive variables
#        cdef DoubleArray dl = self.left_state.get_carray("density")
#        cdef DoubleArray pl = self.left_state.get_carray("pressure")
#
#        # right state primitive variables
#        cdef DoubleArray dr = self.right_state.get_carray("density")
#        cdef DoubleArray pr = self.right_state.get_carray("pressure")
#
#        # extract pointers
#        particles.pointer_groups(x, particles.named_groups['position'])
#        particles.pointer_groups(dcx, particles.named_groups['dcom'])
#        particles.pointer_groups(v, particles.named_groups['velocity'])
#
#        self.left_state.pointer_groups(vl,  self.left_state.named_groups['velocity'])
#        self.right_state.pointer_groups(vr, self.right_state.named_groups['velocity'])
#
#        mesh.faces.pointer_groups(fij, mesh.faces.named_groups['com'])
#        mesh.faces.pointer_groups(wx,  mesh.faces.named_groups['velocity'])
#
#        # allocate space and compute gradients
#        self.grad.resize(particles.get_number_of_items())
#        self.compute_gradients(particles, mesh.faces)
#
#        self.grad.pointer_groups(dd, self.reconstruct_grad_groups['density'])
#        self.grad.pointer_groups(dv, self.reconstruct_grad_groups['velocity'])
#        self.grad.pointer_groups(dp, self.reconstruct_grad_groups['pressure'])
#
#        if do_colors:
#            self.particles(self.col, particles.named_groups["colors"])
#            self.left_state.pointer_groups(self.coll,  self.named_groups['colors'])
#            self.right_state.pointer_groups(self.colr, self.named_groups['colors'])
#            self.grad.pointer_groups(dc, self.reconstruct_grad_groups['colors'])
#
#        # create left/right states for each face
#        for m in range(mesh.faces.get_number_of_items()):
#
#            i = pair_i.data[m]
#            j = pair_j.data[m]
#
#            # density
#            dl.data[m] = d.data[i]
#            dr.data[m] = d.data[j]
#
#            if do_colors:
#                for c in range(num_colors):
#                    self.coll[c][m] = self.col[c][i]
#                    self.colr[c][m] = self.col[c][j]
#
#            # pressure
#            pl.data[m] = p.data[i]
#            pr.data[m] = p.data[j]
#
#            # velocity - add pressure gradient component
#            for k in range(dim):
#
#                # copy velocities
#                if boost:
#                    vi[k] = v[k][i] - wx[k][m]
#                    vj[k] = v[k][j] - wx[k][m]
#                else:
#                    vi[k] = v[k][i]
#                    vj[k] = v[k][j]
#
#                vl[k][m] = vi[k] - fac*dp[k][i]/d.data[i]
#                vr[k][m] = vj[k] - fac*dp[k][j]/d.data[j]
#
#            # MUSCL schemem eq. 36
#            for k in range(dim): # dot products
#
#                # distance from particle to com of face
#                sepi = fij[k][m] - (x[k][i] + dcx[k][i])
#                sepj = fij[k][m] - (x[k][j] + dcx[k][j])
#
#                # add gradient (eq. 21) and time extrapolation (eq. 37)
#                # the trace of dv is div of velocity
#                dl.data[m] += dd[k][i]*(sepi - fac*vi[k]) - fac*d.data[i]*dv[(dim+1)*k][i]
#                dr.data[m] += dd[k][j]*(sepj - fac*vj[k]) - fac*d.data[j]*dv[(dim+1)*k][j]
#
#                if do_colors:
#                    for c in range(num_colors):
#                        self.coll[c][m] += self.dc[c*num_colors+k]*(sepi - fac*vi[k]) -\
#                                fac*self.col[c][i]*dv[(dim+1)*k][i]
#                        self.colr[c][m] += self.dc[c*num_colors+k]*(sepj - fac*vj[k]) -\
#                                fac*self.col[c][j]*dv[(dim+1)*k][j]
#
#                pl.data[m] += dp[k][i]*(sepi - fac*vi[k]) - fac*gamma*p.data[i]*dv[(dim+1)*k][i]
#                pr.data[m] += dp[k][j]*(sepj - fac*vj[k]) - fac*gamma*p.data[j]*dv[(dim+1)*k][j]
#
#                # pressure term already added before loop 
#                for n in range(dim): # over velocity components
#                    vl[n][m] += dv[n*dim+k][i]*(sepi - fac*vi[k])
#                    vr[n][m] += dv[n*dim+k][j]*(sepj - fac*vj[k])
#
#            if dl.data[m] <= 0.0:
#                raise RuntimeError('left density less than zero...... id: %d (%f, %f)' %(i, x[0][i], x[1][i]))
#            if dr.data[m] <= 0.0:
#                raise RuntimeError('right density less than zero..... id: %d (%f, %f)' %(j, x[0][j], x[1][j]))
#            if pl.data[m] <= 0.0:
#                raise RuntimeError('left pressure less than zero..... id: %d (%f, %f)' %(i, x[0][i], x[1][i]))
#            if pr.data[m] <= 0.0:
#                raise RuntimeError('right pressure less than zero.... id: %d (%f, %f)' %(j, x[0][j], x[1][j]))


# ---------------------------------- color fields functions ----------------------------------

#cdef class ReconstructionBase:
    #def set_fields_to_reconstruct(CarrayContainer particles):
    #    msg = "Reconstruction::set_fields_to_reconstruct called!"
    #    raise NotImplementedError(msg)

#cdef class PieceWiseConstant(ReconstructionBase):
#    def __dealloc__(self):
#        """Release pointers"""
#
#        if self.do_colors:
#            stdlib.free(self.col)
#            stdlib.free(self.coll)
#            stdlib.free(self.colr)
#
#    def set_fields_to_reconstruct(CarrayContainer particles):
        # add species
#        if "species" in particles.named_groups.keys():
#            for field in particles.named_groups["species"]:
#                named_groups["primitive"].append(field)
#                named_groups["species"].append(field)
#                state_vars[field] = "double"
#
#        # add passive-scalars
#        if "passive-scalars" in particles.named_groups.keys():
#            for field in particles.named_groups["passive-scalars"]:
#                named_groups["primitive"].append(field)
#                named_groups["passive-scalars"].append(field)
#                state_vars[field] = "double"
#
#        # create extra groups for convenience
#        if "species" in particles.named_groups.keys() or\
#                "passive-scalars" in particles.named_groups.keys():
#            self.do_colors = True
#    def initialize(self):
#        """Setup initial arrays and routines for computation"""
#        if self.reconstruct_fields or self.reconstruct_field_groups:
#            raise RuntimeError("fields to reconstruct not specified")
#
#        # create states
#        self.left_state  = CarrayContainer(var_dict=self.reconstruct_fields)
#        self.right_state = CarrayContainer(var_dict=self.reconstruct_fields)
#
#        if self.do_colors:
#            num_colors = len(particles.named_groups["colors"])
#            self.col  = <np.float64_t**> stdlib.malloc(num_colors*sizeof(void*))
#            self.coll = <np.float64_t**> stdlib.malloc(num_colors*sizeof(void*))
#            self.colr = <np.float64_t**> stdlib.malloc(num_colors*sizeof(void*))
#    cpdef compute_states(self, CarrayContainer particles, Mesh mesh, EquationStateBase eos,
#            RiemannBase riemann, DomainManager domain_manager, double dt, int dim):
#
#        cdef int i, j, k, n#, c, num_colors
#        cdef bint do_colors = self.do_colors
#        if do_colors:
#            num_colors = len(self.named_groups["colors"])
#            particles.pointer_groups(self.col, self.particles.named_groups["colors"])
#            self.left_state.pointer_groups(self.coll, self.particles.named_groups["colors"])
#            self.right_state.pointer_groups(self.colr, self.particles.named_groups["colors"])
#
#            if do_colors:
#                for c in range(num_colors):
#                    self.coll[c][m] = self.col[c][i]
#                    self.colr[c][m] = self.col[c][j]
