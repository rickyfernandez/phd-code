import logging
import numpy as np

from ..utils.tools import check_class

from ..mesh.mesh import Mesh
from ..boundary.boundary import Boundary
from ..domain.domain import DomainLimits
from ..riemann.riemann import RiemannBase
from ..containers.containers import CarrayContainer
from ..reconstruction.reconstruction import ReconstructionBase

phdLogger = logging.getLogger('phd')

callbacks = []

class Eos(object):
    pass

class NewIntegrateBase(object):
    def __init__(self, time=0, final_time=1.0, iteration=0,
            max_iteration=100000,**kwargs):
        """Constructor for the Integrator"""
        self.load_balance = None
        self.particles = self.eos = None
        self.reconstruction = self.riemann = None
        self.mesh = self.domain = self.boundary = None

        # simulation time
        self.time = time
        self.final_time = final_time

        # for paralel runs
        self.is_parallel = False
        self.loc_dt = np.zeros(1)
        self.glb_dt = np.zeros(1)

        # integration iteration number
        self.iteration = iteration
        self.max_iteration = max_iteration

    def initialize_and_link(self):
        """
        """
        self._check_component()

        # add load balance if in parallel 
        self.domain_manager = DomainManager(self.is_paralell)
        if self.is_parallel:

            # load balance
            self.load_balance = LoadBalance()
            self.load_balance.comm = self.comm
            self.load_balance.domain = self.domain
            self.load_balance._initialize()

            # boundary
            self.boundary.comm = self.comm
            self.boundary.load_bal = self.load_balance

        # boundary class
        self.boundary.domain = self.domain
        self.boundary._initialize()

        # mesh class
        self.mesh.boundary = self.boundary
        self.mesh._initialize()

        # reconstruction class
        self.reconstruction.mesh = self.mesh
        self.reconstruction.particles = self.particles
        self.reconstruction._initialize()

        # riemann class
        self.riemann.reconstruction = self.reconstruction

    def compute_time_step(self):
        '''
        Compute time step for current state of the simulation.
        Works in serial and parallel.
        '''
        self.loc_dt[0] = self.riemann.compute_time_step()

        # if in parallel find smallest dt across
        # all meshes 
        if self.is_parallel:
            self.comm.Allreduce(sendbuf=local_dt,
                    recvbuf=global_dt, op=MPI.MIN)
            return self.glb_dt[0]
        return self.loc_dt[0]

    def finished(self):
        '''
        Check if the simulation has reached its final time or
        reached max iteration number.
        '''
        return self.time >= self.final_time or\
                self.iteration >= self.max_iteration

    def evolve_timestep(self):
        msg = "IntegrateBase::evolve_timestep called!"
        raise NotImplementedError(msg)

    def set_parallel(self, is_parallel):
        '''Set parallel flag'''
        self.is_parallel = is_parallel

    def set_comm(self, comm):
        '''Set mpi communicator'''
        self.comm = comm

    @check_class(Eos)
    def set_eos(self, cl):
        '''Set equation of state for gas'''
        self.equation_state = cl

    @check_class(CarrayContainer)
    def set_particles(self, cl):
        '''Set particles to simulate'''
        self.particles = cl

    @check_class(DomainLimits)
    def set_domain(self, cl):
        '''Set spatial domain exten'''
        self.domain = cl

    @check_class(Boundary)
    def set_boundary(self, cl):
        '''Set boundary generator'''
        self.boundary = cl

    @check_class(Mesh)
    def set_mesh(self, cl):
        '''Set spatial mesh'''
        self.mesh = cl

    @check_class(ReconstructionBase)
    def set_reconstruction(self, cl):
        '''Set reconstruction method'''
        self.reconstruction = cl

    @check_class(RiemannBase)
    def set_riemann(self, cl):
        '''Set riemann solver'''
        self.riemann = cl

class StaticMesh(NewIntegrateBase):
    '''
    Static mesh integrator. Once the mesh is created in `begin_loop` method
    the mesh will stay static throughout the simulation.
    '''
    def __init__(self, time=0, final_time=1.0, iteration=0,
            max_iteration=100000,**kwargs):

        # call super
        super(StaticMesh, self).__init__(time, final_time,
                iteration, max_iteration,**kwargs)

    def begin_loop(self, simulation):
        '''
        Build initial mesh, the mesh is only built once.
        '''

        # ignored if in serial 
        self.domain_manager.parition()

        # build mesh and geometric quantities
        self.mesh.build_geometry(self.particles)

        # relax mesh if needed
        if self.mesh.relax_mesh and not simulation.restart:
            for i in range(self.mesh.num_relax_iteration):

                # lloyd relaxation
                self.mesh.relax(self.particles)

                # save data if requested
                if simulation.save_mesh_relaxation:
                    simulation.output_data()

        # compute density, velocity, pressure, ...
        self.equation_state.compute_primitive(self.particles)

        # assign velocities to mesh cells and faces 
        self.mesh.assign_mesh_generator_velocities(self.particles)
        self.mesh.assign_face_velocities()

    def evolve_timestep(self):
        '''
        Solve the compressible gas equations
        '''
        phdLogger.info('Begining integration...')

        # solve the riemann problem at each face
        self.reconstruction.compute_states(self.particles)
        self.riemann.compute_fluxes(self.particles, self.reconstruction)
        self.mesh.update_from_fluxes(self.particles, self.riemann)

        # setup the mesh for the next setup 
        self.eos.primitive_from_conserative(self.particles)
        self.iteration_count += 1; current_time += dt

        phdLogger.info('Finished integration')

    def after_loop(self):
        pass

class MovingMesh(StaticMesh):
    '''
    Moving mesh integrator.
    '''
    def __init__(self, time=0, final_time=1.0, iteration=0,
            max_iteration=100000,**kwargs):

        # call super
        super(StaticMesh, self).__init__(time, final_time,
                iteration, max_iteration,**kwargs)

    def evolve_timestep(self):
        '''
        Solve the compressible gas equations
        '''
        phdLogger.info('Begining integration...')

        # ignored if serial run
        if self.domain_manager.check_for_partion():
            self.domain_manager.partion()

        # assign velocities to mesh cells and faces 
        self.mesh.assign_mesh_generator_velocities(self.particles)
        self.mesh.assign_face_velocities()

        # solve the riemann problem at each face
        self.reconstruction.compute_states(self.particles)
        self.riemann.compute_fluxes(self.particles, self.reconstruction)
        self.mesh.update_from_fluxes(self.particles, self.riemann)

        # update mesh generator positions
        self.mesh.move_generators(self.particles)
        self.domain_manager.migrate_boundary_particles(self.particles)

        # setup the mesh for the next setup 
        self.mesh.build_geometry(self.particles)
        self.eos.primitive_from_conserative(self.particles)

        self.iteration_count += 1; current_time += dt
        phdLogger.info('Finished integration')
