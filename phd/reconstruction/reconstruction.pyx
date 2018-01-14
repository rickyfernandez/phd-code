import logging
import numpy as np

cimport numpy as np
cimport libc.stdlib as stdlib
from libc.math cimport sqrt, fmax, fmin

from ..utils.particle_tags import ParticleTAGS
from ..utils.carray cimport DoubleArray, IntArray, LongArray

phdLogger = logging.getLogger("phd")

cdef int REAL = ParticleTAGS.Real

cdef class ReconstructionBase:
    def __init__(self, **kwargs):
        self.fields_registered = False
        self.has_passive_scalars = False

    def initialize(self):
        """Setup all connections for computation classes. Should always
        check if fields_registered is True.
        """
        msg = "Reconstruction::initialize called!"
        raise NotImplementedError(msg)

    def add_fields(self, CarrayContainer particles):
        """Create lists of variables to reconstruct and setup containers
        for gradients and reconstructions.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        msg = "Reconstruction::initialize called!"
        raise NotImplementedError(msg)

    cpdef compute_gradients(self, CarrayContainer particles, Mesh mesh,
                            DomainManager domain_manager):
        """Create spatial derivatives for reconstruction.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        mesh : Mesh
            Class that builds the domain mesh.

        """
        msg = "Reconstruction::_compute_gradients called!"
        raise NotImplementedError(msg)

    cpdef compute_states(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost, bint add_temporal=True):
        """Perform reconstruction from cell center to face center.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        mesh : Mesh
            Class that builds the domain mesh.

        boost : bool
            Solve equations in moving reference frame.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        dt : float
            Time step of the simulation.

        """
        msg = "Reconstruction::compute called!"
        raise NotImplementedError(msg)

cdef class PieceWiseConstant(ReconstructionBase):
    def __init__(self, **kwargs):
        super(PieceWiseConstant, self).__init__(**kwargs)

    def initialize(self):
        """Setup all connections for computation classes. Should always
        check if fields_registered is True.
        """
        if not self.fields_registered:
            raise RuntimeError(
                    "Reconstruction did not set fields to reconstruct!")

        # left/right face states for riemann solver
        self.left_states  = CarrayContainer(carrays_to_register=self.reconstruct_fields)
        self.right_states = CarrayContainer(carrays_to_register=self.reconstruct_fields)

        # named groups for easier selection
        self.left_states.carray_named_groups  = self.reconstruct_field_groups
        self.right_states.carray_named_groups = self.reconstruct_field_groups

    def add_fields(self, CarrayContainer particles):
        """Create lists of variables to reconstruct and setup containers
        for gradients and reconstructions.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef str field_name
        cdef dict carray_to_register = {}, carray_named_groups = {}

        if "primitive" not in particles.carray_named_groups or\
                "velocity" not in particles.carray_named_groups:
                    raise RuntimeError("ERROR: Missing fields in particles!")

        # add primitive fields
        for field_name in particles.carray_named_groups["primitive"]:
            carray_to_register[field_name] = "double"

        # add velocity in named groups
        carray_named_groups["primitive"] = particles.carray_named_groups["primitive"]
        carray_named_groups["velocity"] = particles.carray_named_groups["velocity"]

        # add passive-scalars if any 
        if "passive-scalars" in particles.carray_named_groups.keys():
            self.has_passive_scalars = True
            carray_named_groups["passive-scalars"] = particles.carray_named_groups["passive-scalars"]
            self.num_passive = len(particles.carray_named_groups["passive-scalars"])

        # store fields info
        self.fields_registered = True
        self.reconstruct_fields = carray_to_register
        self.reconstruct_field_groups = carray_named_groups

    cpdef compute_gradients(self, CarrayContainer particles, Mesh mesh,
                            DomainManager domain_manager):
        """Create spatial derivatives for reconstruction.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        mesh : Mesh
            Class that builds the domain mesh.

        """
        pass # no gradients for constant reconstruction

    cpdef compute_states(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost, bint add_temporal=True):
        """Perform reconstruction from cell center to face center.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        mesh : Mesh
            Class that builds the domain mesh.

        boost : bool
            Solve equations in moving reference frame.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        dt : float
            Time step of the simulation.

        """
        # particle primitive variables
        cdef DoubleArray d = particles.get_carray("density")
        cdef DoubleArray p = particles.get_carray("pressure")

        # density 
        cdef DoubleArray dl = self.left_states.get_carray("density")
        cdef DoubleArray dr = self.right_states.get_carray("density")

        # pressure
        cdef DoubleArray pl = self.left_states.get_carray("pressure")
        cdef DoubleArray pr = self.right_states.get_carray("pressure")

        # particle indices that make up face
        cdef LongArray pair_i = mesh.faces.get_carray("pair-i")
        cdef LongArray pair_j = mesh.faces.get_carray("pair-j")

        cdef int i, j, k, n, dim, num_species
        cdef np.float64_t *v[3], *vl[3], *vr[3], *wx[3]

        phdLogger.info("PieceWiseConstant: Starting reconstruction")

        dim = len(particles.carray_named_groups["position"])

        # particle and face velocity pointer
        particles.pointer_groups(v, particles.carray_named_groups["velocity"])
        mesh.faces.pointer_groups(wx, mesh.faces.carray_named_groups["velocity"])

        # resize states to hold values at each face
        self.left_states.resize(mesh.faces.get_carray_size())
        self.right_states.resize(mesh.faces.get_carray_size())

        # velocity
        self.left_states.pointer_groups(vl,  self.left_states.carray_named_groups["velocity"])
        self.right_states.pointer_groups(vr, self.right_states.carray_named_groups["velocity"])

        if self.has_passive_scalars:

            particles.pointer_groups(self.passive, particles.carray_named_groups["passive-scalars"])
            self.left_states.pointer_groups(self.passive_l, self.carray_named_groups["passive-scalars"])
            self.right_states.pointer_groups(self.passive_r, self.carray_named_groups["passive-scalars"])

        # loop through each face
        for n in range(mesh.faces.get_carray_size()):

            # particles that make up the face
            i = pair_i.data[n]
            j = pair_j.data[n]

            # density 
            dl.data[n] = d.data[i]
            dr.data[n] = d.data[j]

            if self.has_passive_scalars:
                for k in range(self.num_passive):
                    self.passive_l[k][n] = self.passive[k][i]
                    self.passive_r[k][n] = self.passive[k][j]

            # pressure
            pl.data[n] = p.data[i]
            pr.data[n] = p.data[j]

            # velocities
            for k in range(dim):
                if boost:
                    vl[k][n] = v[k][i] - wx[k][n]
                    vr[k][n] = v[k][j] - wx[k][n]
                else:
                    vl[k][n] = v[k][i]
                    vr[k][n] = v[k][j]


cdef class PieceWiseLinear(ReconstructionBase):
    def __init__(self, int limiter = 0, **kwargs):
        super(PieceWiseLinear, self).__init__(**kwargs)
        self.limiter = limiter

    def __dealloc__(self):
        """Release pointers"""

        stdlib.free(self.prim_pointer)
        stdlib.free(self.grad_pointer)

        if self.has_passive_scalars:
            stdlib.free(self.passive)
            stdlib.free(self.passive_l)
            stdlib.free(self.passive_r)
            stdlib.free(self.dpassive)

        stdlib.free(self.phi_max)
        stdlib.free(self.phi_min)

        stdlib.free(self.alpha)
        stdlib.free(self.df)

    def add_fields(self, CarrayContainer particles):
        """
        Create lists of variables to reconstruct and setup containers for
        gradients and reconstructions
        """
        cdef int i, dim
        cdef str field_name, grad_name
        cdef list axis = ["x", "y", "z"]
        cdef dict grad_carray_to_register = {}, grad_carray_named_groups = {}
        cdef dict recon_carray_to_register = {}, recon_carray_named_groups = {}

        if "primitive" not in particles.carray_named_groups or\
                "velocity" not in particles.carray_named_groups:
                    raise RuntimeError("ERROR: Missing fields in particles!")

        dim = len(particles.carray_named_groups["position"])

        grad_carray_named_groups["velocity"] = []
        grad_carray_named_groups["primitive"] = []
        for field_name in particles.carray_named_groups["primitive"]:
            grad_carray_named_groups[field_name] = []

        recon_carray_named_groups["primitive"] = particles.carray_named_groups["primitive"]

        # add primitive fields
        for field_name in particles.carray_named_groups["primitive"]:

            # add field to primitive reconstruction fields
            recon_carray_to_register[field_name] = "double"
            for i in range(dim):

                # store gradient of field
                grad_name = field_name + "_" + axis[i]
                grad_carray_to_register[grad_name] = "double"
                grad_carray_named_groups["primitive"].append(grad_name)
                grad_carray_named_groups[field_name].append(grad_name)

                # store velocity gradient matrix
                if "vel" in field_name:
                    grad_carray_named_groups["velocity"].append(grad_name)

        # store velocity group
        recon_carray_named_groups["velocity"] = particles.carray_named_groups["velocity"]

        # add passive-scalars if any 
        if "passive-scalars" in particles.carray_named_groups.keys():
            self.has_passive_scalars = True
            self.num_passive = len(particles.carray_named_groups["passive-scalars"])

            grad_carray_named_groups["passive-scalars"] = []
            recon_carray_named_groups["passive-scalars"] = particles.carray_named_groups["passive-scalars"]

            for field_name in particles.carray_named_groups["passive-scalars"]:
                for i in range(dim):
                    grad_name = field_name + "_" + axis[i]
                    grad_carray_named_groups["passive-scalars"].append(grad_name)

        # store fields
        self.fields_registered = True
        self.reconstruct_fields = recon_carray_to_register
        self.reconstruct_field_groups = recon_carray_named_groups

        # store gradients
        self.reconstruct_grads = grad_carray_to_register
        self.reconstruct_grad_groups = grad_carray_named_groups

    def initialize(self):
        """Setup initial arrays and routines for computation."""
        if not self.fields_registered:
            raise RuntimeError(
                    "Reconstruction did not set fields to reconstruct!")

        # initialize left/right face states for riemann solver
        self.left_states  = CarrayContainer(carrays_to_register=self.reconstruct_fields)
        self.right_states = CarrayContainer(carrays_to_register=self.reconstruct_fields)

        # add named groups
        self.left_states.carray_named_groups  = self.reconstruct_field_groups
        self.right_states.carray_named_groups = self.reconstruct_field_groups

        self.grad = CarrayContainer(carrays_to_register=self.reconstruct_grads)
        self.grad.carray_named_groups = self.reconstruct_grad_groups

        # allocate helper pointers
        dim = len(self.left_states.carray_named_groups["velocity"])
        num_fields = len(self.left_states.carray_named_groups["primitive"])

        if self.has_passive_scalars:
            self.num_passive = len(self.fields_to_reconstruct_groups["passive_scalars"])
            self.passive_l = <np.float64_t**> stdlib.malloc(self.num_passive*sizeof(void*))
            self.passive_r = <np.float64_t**> stdlib.malloc(self.num_passive*sizeof(void*))
            self.dpassive  = <np.float64_t**> stdlib.malloc((self.num_passive*dim)*sizeof(void*))

        # primitive values and gradient
        self.prim_pointer = <np.float64_t**> stdlib.malloc(num_fields*sizeof(void*))
        self.grad_pointer = <np.float64_t**> stdlib.malloc((num_fields*dim)*sizeof(void*))

        # min/max of field value of particle
        self.phi_max = <np.float64_t*> stdlib.malloc(num_fields*sizeof(np.float64))
        self.phi_min = <np.float64_t*> stdlib.malloc(num_fields*sizeof(np.float64))
        self.alpha   = <np.float64_t*> stdlib.malloc(num_fields*sizeof(np.float64))

        # difference of field value at paticle position to face position
        self.df = <np.float64_t*> stdlib.malloc((num_fields*dim)*sizeof(np.float64))

    cpdef compute_gradients(self, CarrayContainer particles, Mesh mesh,
                            DomainManager domain_manager):
        """Compute gradients for each primitive variable.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        mesh : Mesh
            Class that builds the domain mesh.
        """
        # particle information
        cdef IntArray tags = particles.get_carray("tag")
        cdef DoubleArray vol = particles.get_carray("volume")

        cdef DoubleArray face_area = mesh.faces.get_carray("area")
        cdef LongArray pair_i = mesh.faces.get_carray("pair-i")
        cdef LongArray pair_j = mesh.faces.get_carray("pair-j")

        cdef int dim, num_fields
        cdef int limiter = self.limiter

        cdef int i, j, k, n, m, fid
        cdef double dph, psi, d_dif, d_sum

        cdef double *x[3], *dcx[3]
        cdef double cfx[3], *fij[3], area
        cdef double xi[3], xj[3], dr[3], cx[3], r, _vol

        cdef np.float64_t** prim = self.prim_pointer
        cdef np.float64_t** grad = self.grad_pointer

        cdef np.float64_t* phi_max = self.phi_max
        cdef np.float64_t* phi_min = self.phi_min
        cdef np.float64_t* alpha   = self.alpha
        cdef np.float64_t* df      = self.df

        phdLogger.info("PieceWiseLinear: Starting gradient cacluation")
        self.grad.resize(particles.get_carray_size())

        dim = len(particles.carray_named_groups["position"])
        num_fields = len(particles.carray_named_groups["primitive"])

        # pointer to particle information
        particles.pointer_groups(x, particles.carray_named_groups["position"])
        particles.pointer_groups(dcx, particles.carray_named_groups["dcom"])
        particles.pointer_groups(prim, particles.carray_named_groups["primitive"])
        #raise RuntimeError("Hello World")


        # pointer to face center of mass
        mesh.faces.pointer_groups(fij, mesh.faces.carray_named_groups["com"])

        # pointer to primitive gradients with dimension stacked
        self.grad.pointer_groups(grad, self.grad.carray_named_groups["primitive"])

        # calculate gradients
        for i in range(particles.get_carray_size()):
            if tags.data[i] == REAL:

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
                        df[dim*n+k] = 0

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
                        raise RuntimeError("ERROR: incorrect neighbors!")

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
                            df[dim*n+k] += area*(d_dif*cfx[k] - 0.5*d_sum*dr[k])/(r*_vol)

                # limit gradients eq. 30
                for m in range(mesh.neighbors[i].size()):

                    # index of face neighbor
                    fid = mesh.neighbors[i][m]

                    if limiter == 0: # AREPO limiter

                        for n in range(num_fields):

                            dphi = 0
                            for k in range(dim):
                                dphi += df[dim*n+k]*(fij[k][fid] - cx[k])

                            if dphi > 0.0:
                                psi = (phi_max[n] - prim[n][i])/dphi
                            elif dphi < 0.0:
                                psi = (phi_min[n] - prim[n][i])/dphi
                            else:
                                psi = 1.0

                            alpha[n] = fmin(alpha[n], psi)

                    elif limiter == 1: # TESS limiter

                        for n in range(num_fields):

                            # extract neighbor from face
                            if i == pair_i.data[fid]:
                                j = pair_j.data[fid]
                            elif i == pair_j.data[fid]:
                                j = pair_i.data[fid]

                            dphi = 0
                            for k in range(dim):
                                dphi += df[dim*n+k]*(fij[k][fid] - cx[k])

                            if dphi > 0.0:
                                psi = max((prim[n][j] - prim[n][i])/dphi, 0.)
                            elif dphi < 0.0:
                                psi = max((prim[n][j] - prim[n][i])/dphi, 0.)
                            else:
                                psi = 1.0

                            alpha[n] = fmin(alpha[n], psi)

                # store the gradients
                for n in range(num_fields):
                    for k in range(dim):
                        grad[dim*n+k][i] = alpha[n]*df[dim*n+k]

        # transfer gradients to ghost particles
        domain_manager.update_ghost_gradients(particles, self.grad)

    cpdef compute_states(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost, bint add_temporal=True):
        """Perform reconstruction from cell center to face center.

        This follows the method outlined by Springel (2009) and all equations
        referenced are from that paper. The method performs a linear reconstruction
        of the primitive variables by adding a spatial and time derivative.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        mesh : Mesh
            Class that builds the domain mesh.

        boost : bool
            Solve equations in moving reference frame.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        dt : float
            Time step of the simulation.

        """
        cdef double sepi, sepj
        cdef double fac = 0.5*dt #TO FIX: take in the rigth dt
        cdef int i, j, k, m, n, dim, num_passive

        cdef np.float64_t vi[3], vj[3]
        cdef np.float64_t *fij[3], *wx[3]
        cdef np.float64_t *vl[3], *vr[3]
        cdef np.float64_t *x[3], *v[3], *dcx[3]
        cdef np.float64_t *dd[3], *dv[9], *dp[3]

        cdef LongArray pair_i = mesh.faces.get_carray("pair-i")
        cdef LongArray pair_j = mesh.faces.get_carray("pair-j")

        # particle primitive variables
        cdef DoubleArray d = particles.get_carray("density")
        cdef DoubleArray p = particles.get_carray("pressure")

        # left state primitive variables
        cdef DoubleArray dl = self.left_states.get_carray("density")
        cdef DoubleArray pl = self.left_states.get_carray("pressure")

        # right state primitive variables
        cdef DoubleArray dr = self.right_states.get_carray("density")
        cdef DoubleArray pr = self.right_states.get_carray("pressure")

        phdLogger.info("PieceWiseLinear: Starting reconstruction")
        dim = len(particles.carray_named_groups["position"])

        # resize states to hold values at each face
        self.left_states.resize(mesh.faces.get_carray_size())
        self.right_states.resize(mesh.faces.get_carray_size())

        # extract pointers
        particles.pointer_groups(x, particles.carray_named_groups["position"])
        particles.pointer_groups(dcx, particles.carray_named_groups["dcom"])
        particles.pointer_groups(v, particles.carray_named_groups["velocity"])

        self.left_states.pointer_groups(vl,  self.left_states.carray_named_groups["velocity"])
        self.right_states.pointer_groups(vr, self.right_states.carray_named_groups["velocity"])

        mesh.faces.pointer_groups(fij, mesh.faces.carray_named_groups["com"])
        mesh.faces.pointer_groups(wx,  mesh.faces.carray_named_groups["velocity"])

        self.grad.pointer_groups(dd, self.grad.carray_named_groups["density"])
        self.grad.pointer_groups(dv, self.grad.carray_named_groups["velocity"])
        self.grad.pointer_groups(dp, self.grad.carray_named_groups["pressure"])

        if self.has_passive_scalars:

            num_passive = self.num_passive
            particles.pointer_groups(self.passive, particles.carray_named_groups["passive-scalars"])

            self.left_states.pointer_groups(self.passive_l, self.carray_named_groups["passive-scalars"])
            self.right_states.pointer_groups(self.passive_r, self.carray_named_groups["passive-scalars"])

            self.grad.pointer_groups(self.dpassive, self.reconstruct_grad_groups["passive-scalars"])

        # create left/right states for each face
        for m in range(mesh.faces.get_carray_size()):

            # particles that make up the face
            i = pair_i.data[m]
            j = pair_j.data[m]

            # density
            dl.data[m] = d.data[i]
            dr.data[m] = d.data[j]

            if self.has_passive_scalars:
                for k in range(num_passive):

                    # passive-scalars
                    self.passive_l[k][m] = self.passive[k][i]
                    self.passive_r[k][m] = self.passive[k][j]

            # pressure
            pl.data[m] = p.data[i]
            pr.data[m] = p.data[j]

            # velocity
            for k in range(dim):

                # copy velocities for temporal calculation
                if boost:
                    vi[k] = v[k][i] - wx[k][m]
                    vj[k] = v[k][j] - wx[k][m]
                else:
                    vi[k] = v[k][i]
                    vj[k] = v[k][j]

                if add_temporal:
                    # velocity, add time derivative
                    vl[k][m] = vi[k] - fac*dp[k][i]/d.data[i]
                    vr[k][m] = vj[k] - fac*dp[k][j]/d.data[j]
                else:
                    vl[k][m] = vi[k]
                    vr[k][m] = vj[k]

            # add derivatives to primitive 
            for k in range(dim): # dot products

                # distance from particle to com of face
                sepi = fij[k][m] - (x[k][i] + dcx[k][i])
                sepj = fij[k][m] - (x[k][j] + dcx[k][j])

                # add gradient (eq. 21) and time extrapolation (eq. 37)
                # the trace of dv is div of velocity

                # density, add spatial derivative
                dl.data[m] += dd[k][i]*(sepi - fac*vi[k])
                dr.data[m] += dd[k][j]*(sepj - fac*vj[k])

                if add_temporal:
                    # density, add time derivative
                    dl.data[m] -= fac*d.data[i]*dv[(dim+1)*k][i]
                    dr.data[m] -= fac*d.data[j]*dv[(dim+1)*k][j]

                if self.has_passive_scalars:
                    for n in range(num_passive):

                        # passive scalars, add spatial derivative
                        self.passive_l[n][m] += self.dpassive[n*num_passive + k][m]*(sepi - fac*vi[k])
                        self.passive_r[n][m] += self.dpassive[n*num_passive + k][m]*(sepj - fac*vj[k])

                        if add_temporal:
                            # passive scalars, add time derivative
                            self.passive_l[n][m] -= fac*self.passive[n][i]*dv[(dim+1)*k][i]
                            self.passive_r[n][m] -= fac*self.passive[n][j]*dv[(dim+1)*k][j]

                # pressure, add spatial derivative
                pl.data[m] += dp[k][i]*(sepi - fac*vi[k])
                pr.data[m] += dp[k][j]*(sepj - fac*vj[k])

                if add_temporal:
                    # pressure, add time derivative
                    pl.data[m] -= fac*gamma*p.data[i]*dv[(dim+1)*k][i]
                    pr.data[m] -= fac*gamma*p.data[j]*dv[(dim+1)*k][j]

                # velocity, add spatial derivative
                for n in range(dim): # over velocity components
                    vl[n][m] += dv[n*dim+k][i]*(sepi - fac*vi[k])
                    vr[n][m] += dv[n*dim+k][j]*(sepj - fac*vj[k])

            if dl.data[m] <= 0.0:
                raise RuntimeError('left density less than zero...... id: %d (%f, %f)' %(i, x[0][i], x[1][i]))
            if dr.data[m] <= 0.0:
                raise RuntimeError('right density less than zero..... id: %d (%f, %f)' %(j, x[0][j], x[1][j]))
            if pl.data[m] <= 0.0:
                raise RuntimeError('left pressure less than zero..... id: %d (%f, %f)' %(i, x[0][i], x[1][i]))
            if pr.data[m] <= 0.0:
                raise RuntimeError('right pressure less than zero.... id: %d (%f, %f)' %(j, x[0][j], x[1][j]))
