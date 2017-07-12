import numpy as np


from ..mesh.mesh cimport Mesh
from ..riemann.riemann cimport RiemannBase
from ..utils.particle_tags import ParticleTAGS
from ..containers.containers cimport CarrayContainer
from ..utils.carray cimport DoubleArray, IntArray, LongLongArray, LongArray

callbacks = []

def check_class(cl):
    def wrapper(func):
        def checked(cl_check):
            if not isinstance(cl_check, cl):
                raise RuntimeError("%s component not type %s" %\
                        cl_check.__class__.__name__, cl.__class__.__name__)
            return cl_check
        return checked
    return wrapper

def class IntegrateBase(object):
    def __init__(self, **kwargs):
        """Constructor for the Integrator"""
        #self.pc = None
        #self.mesh = None
        #self.riemann = None
        self.pc = self.mesh = self.domain = self.riemann = None
        self.boundary = self.integrator = self.reconstruction = None
        self.load_balance = None

        # mesh relaxation
        self.output_relax = output_relax
        self.relax_num_iterations = relax_num_iterations

    def _initialize(self):
        """Constructor for the Integrator"""

        self.dim = self.mesh.dim
        self.gamma = self.riemann.gamma

        cdef str field
        cdef dict flux_vars = {}
        for field in self.pc.named_groups['conserative']:
            flux_vars[field] = 'double'
        self.flux = CarrayContainer(var_dict=flux_vars)
        self.flux.named_groups['momentum'] = self.pc.named_groups['momentum']

        cdef dict state_vars = {}
        for field in self.pc.named_groups['primitive']:
            state_vars[field] = 'double'

        self.left_state  = CarrayContainer(var_dict=state_vars)
        self.left_state.named_groups['velocity'] = self.pc.named_groups['velocity']

        self.right_state = CarrayContainer(var_dict=state_vars)
        self.right_state.named_groups['velocity'] = self.pc.named_groups['velocity']

    def _initialize_and_link(self):
        """
        """
        self._check_component()

        # load balance for parallel runs
        if self.parallel_run:
            self.load_balance.comm = self.comm
            self.load_balance.domain = self.domain
            self.load_balance._initialize()

        # spatial boundary conditions
        if self.parallel_run:
            self.boundary.comm = self.comm
            self.boundary.load_bal = self.load_balance
        self.boundary.domain = self.domain
        self.boundary._initialize()

        self.mesh.boundary = self.boundary
        self.mesh._initialize()

        self.reconstruction.pc = self.pc
        self.reconstruction.mesh = self.mesh
        self.reconstruction._initialize()

        self.riemann.reconstruction = self.reconstruction

        self.integrator.pc = self.pc
        self.integrator.mesh = self.mesh
        self.integrator.riemann = self.riemann
        self.integrator._initialize()

        self.gamma = self.integrator.riemann.gamma
        self.dimensions = 'xyz'[:self.mesh.dim]

    def finished(self):
        return self.time >= self.final_time or self.iteration >= self.max_iteration

    def compute_time_step(self):
        #local_dt  = np.zeros(1)
        #global_dt = np.zeros(1)

        self.local_dt[0] = self.cfl*self.integrator.compute_time_step()

        if self.parallel_run:
            self.comm.Allreduce(sendbuf=local_dt, recvbuf=global_dt, op=MPI.MIN)
            return self.global_dt[0]
        return self.local_dt[0]

    def evolve_timestep(self):
        msg = "IntegrateBase::evolve_timestep called!"
        raise NotImplementedError(msg)

    @check_class(phd.CarrayContainer)
    def add_particles(self, cl):
        self.pc = cl

    @check_class(phd.DomainLimits)
    def add_domain(self, cl):
        self.domain = cl

    @check_class(phd.Boundary)
    def add_boundary(self, cl):
        self.boundary = cl

    @check_class(phd.Mesh)
    def add_mesh(self, cl):
        self.mesh = cl

    @check_class(phd.ReconstructionBase)
    def add_reconstruction(self, cl):
        self.reconstruction = cl

    @check_class(phd.RiemannBase)
    def add_riemann(self, cl):
        self.riemann = cl

    @check_class(phd.IntegrateBase)
    def add_integrator(self, cl):
        self.integrator = cl


class StaticMesh(IntegrateBase):
    def __init__(self, int regularize = 0, double eta = 0.25, **kwargs):
        """Constructor for the Integrator"""

        super(StaticMesh, self).__init__(**kwargs)

    def begin_loop(self):

        # initial load balance
        self.domain_manager.parition()
        self.mesh.build_geometry()

        # assign velocities to mesh cells and faces 
        self.mesh.assign_mesh_generator_velocities()
        self.mesh.assign_face_velocities()

    def after_loop(self):
        pass

    def evolve_timestep(self):

        logger.info('Begining iteration')

        # solve the riemann problem at each face
        self.reconstruction.compute_states()
        self.riemann.compute_fluxes()
        self.mesh.update_from_fluxes()

        # setup the mesh for the next setup 
        self.eos.primitive_from_conserative()
        self.iteration_count += 1; current_time += dt

        logger.info('iteration')

class MovingMesh(IntegrateBase):
    def __init__(self, int regularize = 0, double eta = 0.25, **kwargs):
        """Constructor for the Integrator"""

        super(MovingMesh, self).__init__(**kwargs)
        self.regularize = regularize
        self.eta = eta

    def evolve_timestep(self):

        logger.info('Begining iteration')

        self.domain_manager.parition()

        # assign velocities to mesh cells and faces 
        self.mesh.assign_mesh_generator_velocities()
        self.mesh.assign_face_velocities()

        # solve the riemann problem at each face
        self.reconstruction.compute_states()
        self.riemann.compute_fluxes()
        self.mesh.update_from_fluxes()

        # update mesh generator positions
        self.mesh.move_mesh_generators()
        self.domain_manager.migrate_boundary_particles()

        # setup the mesh for the next setup 
        self.mesh.build_geometry()
        self.eos.primitive_from_conserative()

        self.iteration_count += 1; current_time += dt
        logger.info('iteration')
