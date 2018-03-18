import logging
import numpy as np

cimport numpy as np
cimport libc.stdlib as stdlib
from libc.math cimport sqrt, fmax, fmin, fabs

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

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        msg = "Reconstruction::_compute_gradients called!"
        raise NotImplementedError(msg)

    cpdef add_spatial(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost):
        """Create spatial derivatives for reconstruction.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        mesh : Mesh
            Class that builds the domain mesh.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        msg = "Reconstruction::add_spatial called!"
        raise NotImplementedError(msg)

    cpdef add_temporal(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost):
        """Create spatial derivatives for reconstruction.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        mesh : Mesh
            Class that builds the domain mesh.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        msg = "Reconstruction::add_temporal called!"
        raise NotImplementedError(msg)

    cpdef compute_states(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost):
        """Perform reconstruction from cell center to face center.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        mesh : Mesh
            Class that builds the domain mesh.

        gamma : double
            Ratio of specific heats.

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        dt : float
            Time step of the simulation.

        boost : bool
            Solve equations in moving reference frame.

        add_temporal : bool
            If true add time derivatives in the reconstruction.

        """
        msg = "Reconstruction::compute called!"
        raise NotImplementedError(msg)

cdef class PieceWiseConstant(ReconstructionBase):
    """Reconstruction of primitive variables onto each face using
    constant implementation.

    Attributes
    ----------
    fields_registered : bool
        Flag stating if reconstruction fields have been registered.

    left_states : CarrayContainer
        Left states primitive values for riemann problem.

    right_states : CarrayContainer
        Left states primitive values for riemann problem.

    has_passive_scalars : bool
        Flag indicating if passive scalars are present for
        reconstruction.

    num_passive : int
        Number of passive scalars in particle containers.

    reconstruct_fields : dict
       Dictionary of primitive fields where the keys are the names
       and values are the data type of carrays to create in the
       container.

    reconstruct_field_groups : dict
        Dictionary of collection of field names allowing for ease
        of subsetting of fields.

    """
    def __init__(self, **kwargs):
        super(PieceWiseConstant, self).__init__(**kwargs)

    def __dealloc__(self):
        """Release pointers"""

        if self.has_passive_scalars:
            stdlib.free(self.passive)
            stdlib.free(self.passive_l)
            stdlib.free(self.passive_r)

    def initialize(self):
        """Setup all connections for computation classes. Should always
        check if fields_registered is True.
        """
        cdef int dim

        if not self.fields_registered:
            raise RuntimeError("Reconstruction did not set fields to reconstruct!")

        # left/right face states for riemann solver
        self.left_states  = CarrayContainer(carrays_to_register=self.reconstruct_fields)
        self.right_states = CarrayContainer(carrays_to_register=self.reconstruct_fields)

        # named groups for easier selection
        self.left_states.carray_named_groups  = self.reconstruct_field_groups
        self.right_states.carray_named_groups = self.reconstruct_field_groups

        # allocate helper pointers
        if self.has_passive_scalars:
            dim = len(self.left_states.carray_named_groups["velocity"])
            self.num_passive = len(self.fields_to_reconstruct_groups["passive_scalars"])

            self.passive   = <np.float64_t**> stdlib.malloc(self.num_passive*sizeof(void*))
            self.passive_l = <np.float64_t**> stdlib.malloc(self.num_passive*sizeof(void*))
            self.passive_r = <np.float64_t**> stdlib.malloc(self.num_passive*sizeof(void*))

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

        # add primitive and velocity in named groups
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

        domain_manager : DomainManager
            Class that handels all things related with the domain.

        """
        pass # no gradients for constant reconstruction

    cpdef compute_states(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost):
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
            Time to extrapolate reconstructed fields to.

        add_temporal : bool
            If true add time derivatives in the reconstruction.

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

        # include passive scalars if any
        if self.has_passive_scalars:

            # pointers to passive and left/right states
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

            # add passive scalars if any
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
                    # velocity in face frame
                    vl[k][n] = v[k][i] - wx[k][n]
                    vr[k][n] = v[k][j] - wx[k][n]
                else:
                    vl[k][n] = v[k][i]
                    vr[k][n] = v[k][j]


cdef class PieceWiseLinear(ReconstructionBase):
    """Reconstruction of primitive variables onto each face using
    AREPO implementation (2009).

    Attributes
    ----------
    fields_registered : bool
        Flag stating if reconstruction fields have been registered.

    grad : CarrayContainer
       Gradient of each primitive field.

    left_states : CarrayContainer
        Left states primitive values for riemann problem.

    right_states : CarrayContainer
        Left states primitive values for riemann problem.

    limiter : str
        Value of 0 is AREPOs and 1 is TESS implementation of
        limiting the gradients.

    has_passive_scalars : bool
        Flag indicating if passive scalars are present for
        reconstruction.

    num_passive : int
        Number of passive scalars in particle containers.

    reconstruct_fields : dict
       Dictionary of primitive fields where the keys are the names
       and values are the data type of carrays to create in the
       container.

    reconstruct_field_groups : dict
        Dictionary of collection of field names allowing for ease
        of subsetting of fields.

    reconstruct_grads : dict
       Dictionary of primitive gradients where the keys are the names
       and values are the data type of carrays to create in the
       container.

    reconstruct_grad_groups : dict
        Dictionary of collection of gradient names allowing for ease
        of subsetting of gradients.

    """
    def __init__(self, str limiter = "arepo", bint gizmo_limiter=True, **kwargs):
        super(PieceWiseLinear, self).__init__(**kwargs)

        if limiter == "arepo":
            self.limiter = 0
        elif limiter == "tess":
            self.limiter = 1
        else:
            raise RuntimeError("ERROR: Unrecognized limiter")

        self.gizmo_limiter = gizmo_limiter

    def __dealloc__(self):
        """Release pointers"""

        stdlib.free(self.prim_pointer)

        stdlib.free(self.priml_pointer)
        stdlib.free(self.primr_pointer)
        stdlib.free(self.state_l)
        stdlib.free(self.state_r)

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
        """Create lists of variables to reconstruct and setup containers
        for gradients and reconstructions.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

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
        cdef int dim, num_fields

        if not self.fields_registered:
            raise RuntimeError(
                    "Reconstruction did not set fields to reconstruct!")

        # initialize left/right face states for riemann solver
        self.left_states  = CarrayContainer(carrays_to_register=self.reconstruct_fields)
        self.right_states = CarrayContainer(carrays_to_register=self.reconstruct_fields)

        # add named groups
        self.left_states.carray_named_groups  = self.reconstruct_field_groups
        self.right_states.carray_named_groups = self.reconstruct_field_groups

        # initialize gradients
        self.grad = CarrayContainer(carrays_to_register=self.reconstruct_grads)
        self.grad.carray_named_groups = self.reconstruct_grad_groups

        # allocate helper pointers
        dim = len(self.left_states.carray_named_groups["velocity"])
        num_fields = len(self.left_states.carray_named_groups["primitive"])

        # allocate helper pointers
        if self.has_passive_scalars:
            self.num_passive = len(self.fields_to_reconstruct_groups["passive_scalars"])

            self.passive   = <np.float64_t**> stdlib.malloc(self.num_passive*sizeof(void*))
            self.passive_l = <np.float64_t**> stdlib.malloc(self.num_passive*sizeof(void*))
            self.passive_r = <np.float64_t**> stdlib.malloc(self.num_passive*sizeof(void*))

            self.dpassive  = <np.float64_t**> stdlib.malloc((self.num_passive*dim)*sizeof(void*))

        # primitive values and gradient
        self.prim_pointer = <np.float64_t**> stdlib.malloc(num_fields*sizeof(void*))

        self.state_l = <np.float64_t*> stdlib.malloc(num_fields*sizeof(void*))
        self.state_r = <np.float64_t*> stdlib.malloc(num_fields*sizeof(void*))
        self.priml_pointer = <np.float64_t**> stdlib.malloc(num_fields*sizeof(void*))
        self.primr_pointer = <np.float64_t**> stdlib.malloc(num_fields*sizeof(void*))

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

        # face information
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

                        # gradient estimate Eq. 21
                        for k in range(dim):
                            df[dim*n+k] += area*(d_dif*cfx[k] - 0.5*d_sum*dr[k])/(r*_vol)

                if limiter == 0: # AREPO limiter

                    # limit gradients Eq. 30
                    for n in range(num_fields):
                        for m in range(mesh.neighbors[i].size()):

                            # index of face neighbor
                            fid = mesh.neighbors[i][m]

                            dphi = 0
                            for k in range(dim):
                                dphi += df[dim*n+k]*(fij[k][fid] - cx[k])

                            if dphi > 0:
                                psi = (phi_max[n] - prim[n][i])/dphi
                            elif dphi < 0:
                                psi = (phi_min[n] - prim[n][i])/dphi
                            else:
                                psi = 1.0

                            alpha[n] = fmin(alpha[n], fmax(psi, 0.))

                elif limiter == 1: # TESS limiter

                    # limit gradients Eq. 22
                    for n in range(num_fields):
                        for m in range(mesh.neighbors[i].size()):

                            # index of face neighbor
                            fid = mesh.neighbors[i][m]
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

                            alpha[0] = fmin(alpha[0], fmax(psi, 0.))

                # store the gradients
                for n in range(num_fields):
                    for k in range(dim):
                        grad[dim*n+k][i] = alpha[n]*df[dim*n+k]

        # transfer gradients to ghost particles
        domain_manager.update_ghost_gradients(particles, self.grad)

    cpdef add_spatial(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost):
        """Perform reconstruction from cell center to face center.
        This follows the method outlined by Springel (2009) and all equations
        referenced are from that paper.

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
            Time to extrapolate reconstructed fields to.
        """
        cdef int i, j, k, m, n, dim, num_fields

        # gizmo limiter parameters
        cdef double psi1 = 0.5, psi2 = 0.25
        cdef double delta1, delta2
        cdef double phi_min, phi_max
        cdef double phibar_l, phibar_r
        cdef double phi_minus, phi_plus

        cdef double sepi, sepj
        cdef double sepi_mag, sepj_mag, diff_mag

        cdef np.float64_t *vl[3], *vr[3]
        cdef np.float64_t *fij[3], *wx[3]
        cdef np.float64_t *x[3], *v[3], *dcx[3]

        cdef LongArray pair_i = mesh.faces.get_carray("pair-i")
        cdef LongArray pair_j = mesh.faces.get_carray("pair-j")

        # left state primitive variables
        cdef DoubleArray dl = self.left_states.get_carray("density")
        cdef DoubleArray pl = self.left_states.get_carray("pressure")

        # right state primitive variables
        cdef DoubleArray dr = self.right_states.get_carray("density")
        cdef DoubleArray pr = self.right_states.get_carray("pressure")

        cdef np.float64_t** prim = self.prim_pointer
        cdef np.float64_t** prim_l = self.priml_pointer
        cdef np.float64_t** prim_r = self.primr_pointer

        cdef np.float64_t* state_l = self.state_l
        cdef np.float64_t* state_r = self.state_r

        cdef np.float64_t** grad = self.grad_pointer

        phdLogger.info("PieceWiseLinear: Starting spatial reconstruction")

        dim = len(particles.carray_named_groups["position"])
        num_fields = len(particles.carray_named_groups["primitive"])

        # resize states to hold values at each face
        self.left_states.resize(mesh.faces.get_carray_size())
        self.right_states.resize(mesh.faces.get_carray_size())

        # pointers left/right primitive values
        self.left_states.pointer_groups(prim_l, self.left_states.carray_named_groups["primitive"])
        self.right_states.pointer_groups(prim_r, self.right_states.carray_named_groups["primitive"])

        # pointers to particle primitive, position, com, and velocity
        particles.pointer_groups(prim, particles.carray_named_groups["primitive"])
        particles.pointer_groups(x, particles.carray_named_groups["position"])
        particles.pointer_groups(dcx, particles.carray_named_groups["dcom"])
        particles.pointer_groups(v, particles.carray_named_groups["velocity"])

        # pointers to left/right velocities at face
        self.left_states.pointer_groups(vl, self.left_states.carray_named_groups["velocity"])
        self.right_states.pointer_groups(vr, self.right_states.carray_named_groups["velocity"])

        # pointers to face velocity and center of mass 
        mesh.faces.pointer_groups(fij, mesh.faces.carray_named_groups["com"])
        mesh.faces.pointer_groups(wx, mesh.faces.carray_named_groups["velocity"])

        # pointers to primitive gradients
        self.grad.pointer_groups(grad, self.grad.carray_named_groups["primitive"])

        # create left/right states for each face
        for m in range(mesh.faces.get_carray_size()):

            # particles that make up the face
            i = pair_i.data[m]
            j = pair_j.data[m]

            # copy constant states
            for n in range(num_fields):
                prim_l[n][m] = prim[n][i]
                prim_r[n][m] = prim[n][j]

            # update because of boost
            for k in range(dim):
                if boost:
                    vl[k][m] = v[k][i] - wx[k][m]
                    vr[k][m] = v[k][j] - wx[k][m]

            # copy constant states
            for n in range(num_fields):
                state_l[n] = prim_l[n][m]
                state_r[n] = prim_r[n][m]

            diff_mag = 0.0
            sepi_mag = sepj_mag = 0.0 

            # add spatial derivatives Eq. 27
            for k in range(dim):

                # distance from particle to com of face
                sepi = fij[k][m] - (x[k][i] + dcx[k][i])
                sepj = fij[k][m] - (x[k][j] + dcx[k][j])

                sepi_mag += sepi*sepi
                sepj_mag += sepj*sepj
                diff_mag += (x[k][j] - x[k][i])**2

                # extraploate to face
                for n in range(num_fields):
                    prim_l[n][m] += grad[n*dim+k][i]*sepi
                    prim_r[n][m] += grad[n*dim+k][j]*sepj

            sepi_mag = sqrt(sepi_mag)
            sepj_mag = sqrt(sepj_mag)
            diff_mag = sqrt(diff_mag)

            # gizmo limiter: appendix B4
            if self.gizmo_limiter:

                # limit each field pairwise
                for n in range(num_fields):

                    delta1 = psi1*fabs(state_l[n] - state_r[n])
                    delta2 = psi2*fabs(state_l[n] - state_r[n])

                    phi_min = fmin(state_l[n], state_r[n])
                    phi_max = fmax(state_l[n], state_r[n])

                    phibar_l = state_l[n] + sepi_mag/diff_mag*(state_r[n] - state_l[n])
                    phibar_r = state_r[n] + sepj_mag/diff_mag*(state_l[n] - state_r[n])

                    if ((phi_max + delta1)*phi_max >= 0.):
                        phi_plus = phi_max + delta1
                    else:
                        phi_plus = phi_max/(1 + delta1/fabs(phi_max))

                    if ((phi_min - delta1)*phi_min >= 0.):
                        phi_minus = phi_min - delta1
                    else:
                        phi_minus = phi_min/(1 + delta1/fabs(phi_min))

                    if prim[n][i] < prim[n][j]:
                        prim_l[n][m] = fmax(phi_minus, fmin(phibar_l+delta2, prim_l[n][m]))
                        prim_r[n][m] = fmin(phi_plus,  fmax(phibar_r-delta2, prim_r[n][m]))

                    elif prim[n][i] > prim[n][j]:
                        prim_l[n][m] = fmin(phi_plus,  fmax(phibar_l-delta2, prim_l[n][m]))
                        prim_r[n][m] = fmax(phi_minus, fmin(phibar_r+delta2, prim_r[n][m]))

                    else:
                        prim_l[n][m] = state_l[n]
                        prim_r[n][m] = state_r[n]

                # if negative reduce to constant reconstruction
                if dl.data[m] < 0.0 or pl.data[m] < 0.0:
                    for n in range(num_fields):
                        prim_l[n][m] = state_l[n]

                if dr.data[m] < 0.0 or pr.data[m] < 0.0:
                    for n in range(num_fields):
                        prim_r[n][m] = state_r[n]

    cpdef add_temporal(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost):
        """Perform temporal reconstruction from cell center to face center.
        This follows the method outlined by Springel (2009) and all equations
        referenced are from that paper.

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
            Time to extrapolate reconstructed fields to.
        """
        cdef int i, j, k, m, n, dim, num_passive, num_fields

        cdef np.float64_t vi[3], vj[3]
        cdef np.float64_t *v[3], *wx[3]
        cdef np.float64_t *vl[3], *vr[3]
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

        cdef np.float64_t* state_l = self.state_l
        cdef np.float64_t* state_r = self.state_r

        cdef np.float64_t** prim_l = self.priml_pointer
        cdef np.float64_t** prim_r = self.primr_pointer

        phdLogger.info("PieceWiseLinear: Starting temporal reconstruction")

        dim = len(particles.carray_named_groups["position"])
        num_fields = len(particles.carray_named_groups["primitive"])

        # extract pointers
        particles.pointer_groups(v, particles.carray_named_groups["velocity"])

        # pointers left/right primitive values
        self.left_states.pointer_groups(prim_l, self.left_states.carray_named_groups["primitive"])
        self.right_states.pointer_groups(prim_r, self.right_states.carray_named_groups["primitive"])

        # pointers to velocity states
        self.left_states.pointer_groups(vl, self.left_states.carray_named_groups["velocity"])
        self.right_states.pointer_groups(vr, self.right_states.carray_named_groups["velocity"])

        # pointers to face velocity and center of mass 
        mesh.faces.pointer_groups(wx, mesh.faces.carray_named_groups["velocity"])

        # pointers to gradients
        self.grad.pointer_groups(dd, self.grad.carray_named_groups["density"])
        self.grad.pointer_groups(dv, self.grad.carray_named_groups["velocity"])
        self.grad.pointer_groups(dp, self.grad.carray_named_groups["pressure"])

        if self.has_passive_scalars:

            num_passive = self.num_passive
            particles.pointer_groups(self.passive, particles.carray_named_groups["passive-scalars"])

            # pointer to passive left/right states
            self.left_states.pointer_groups(self.passive_l, self.carray_named_groups["passive-scalars"])
            self.right_states.pointer_groups(self.passive_r, self.carray_named_groups["passive-scalars"])

            # pointer to gradients of passive scalars
            self.grad.pointer_groups(self.dpassive, self.reconstruct_grad_groups["passive-scalars"])

        # create left/right states for each face
        for m in range(mesh.faces.get_carray_size()):

            # particles that make up the face
            i = pair_i.data[m]
            j = pair_j.data[m]

            # copy states before time derivatives
            for n in range(num_fields):
                state_l[n] = prim_l[n][m]
                state_r[n] = prim_r[n][m]

            # velocity
            for k in range(dim):

                # copy velocities for temporal calculation
                if boost:
                    vi[k] = v[k][i] - wx[k][m]
                    vj[k] = v[k][j] - wx[k][m]
                else:
                    vi[k] = v[k][i]
                    vj[k] = v[k][j]

                vl[k][m] -= dt*dp[k][i]/d.data[i]
                vr[k][m] -= dt*dp[k][j]/d.data[j]

            # add derivatives to primitive 
            for k in range(dim): # dot products

                # add gradient (Eq. 21) and time Extrapolation (eq. 37)
                # the trace of dv is div of velocity

                # density, add temporal derivative
                dl.data[m] -= dt*(d.data[i]*dv[(dim+1)*k][i] + vi[k]*dd[k][i])
                dr.data[m] -= dt*(d.data[j]*dv[(dim+1)*k][j] + vj[k]*dd[k][j])

                if self.has_passive_scalars:
                    for n in range(num_passive):

                        # passive scalars, add spatial derivative
                        self.passive_l[n][m] -= dt*(self.passive[n][i]*dv[(dim+1)*k][i]\
                                - vi[k]*self.dpassive[n*dim+k][i])
                        self.passive_r[n][m] -= dt*(self.passive[n][j]*dv[(dim+1)*k][j]\
                                - vj[k]*self.dpassive[n*dim+k][j])

                # pressure, add spatial derivative
                pl.data[m] -= dt*(gamma*p.data[i]*dv[(dim+1)*k][i] + vi[k]*dp[k][i])
                pr.data[m] -= dt*(gamma*p.data[j]*dv[(dim+1)*k][j] + vj[k]*dp[k][j])

                # velocity, add spatial derivative
                for n in range(dim): # over velocity components
                    vl[n][m] -= dt*vi[k]*dv[n*dim+k][i]
                    vr[n][m] -= dt*vj[k]*dv[n*dim+k][j]

            # if negative remove time derivative 
            if dl.data[m] < 0.0 or pl.data[m] < 0.0:
                for n in range(num_fields):
                    prim_l[n][m] = state_l[n]

            if dr.data[m] < 0.0 or pr.data[m] < 0.0:
                for n in range(num_fields):
                    prim_r[n][m] = state_r[n]

    cpdef compute_states(self, CarrayContainer particles, Mesh mesh,
                         double gamma, DomainManager domain_manager,
                         double dt, bint boost):

        self.add_spatial(particles, mesh, gamma, domain_manager, dt, boost)
        self.add_temporal(particles, mesh, gamma, domain_manager, dt, boost)
