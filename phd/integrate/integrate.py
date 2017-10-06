import logging
import numpy as np

import phd
from ..utils.tools import check_class

phdLogger = logging.getLogger('phd')
callbacks = []

class IntegrateBase(object):
    def __init__(self, param_initial_time=0., param_final_time=1.0, param_dim=2):
        """Constructor for Integrate base class. Every integrate class has
        to inherit this class.
        """
        self.param_dim = param_dim
        self.param_final_time = param_final_time
        self.param_initial_time = param_initial_time

        self.dt = 0.                    # time step
        self.iteration = 0              # iteration number
        self.time = param_initial_time  # current time

        # required objects to be set
        self.mesh = None
        self.riemann = None
        self.particles = None
        self.reconstruction = None
        self.domain_manager = None

        # for communication dt across processors
        self.loc_dt = np.zeros(1, dtype=np.float64)
        self.glb_dt = np.zeros(1, dtype=np.float64)

    def initialize(self):
        """
        Setup all connections for computation classes
        """
        msg = "IntegrateBase::initialize called!"
        raise NotImplementedError(msg)

    def compute_time_step(self):
        '''
        Compute time step for current state of the simulation.
        Works in serial and parallel.
        '''
        msg = "IntegrateBase::compute_time_step called!"
        raise NotImplementedError(msg)

    def evolve_timestep(self):
        """
        Evolve the simulation for one time step
        """
        msg = "IntegrateBase::evolve_timestep called!"
        raise NotImplementedError(msg)

    def set_intial_time(self, initial_time):
        '''Set current time'''
        self.param_initial_time = initial_time
        self.time = initial_time

    def set_final_time(self, final_time):
        '''Set current time'''
        self.param_final_time = final_time

    def set_iteration(self, iteration):
        '''Set iteration count'''
        self.iteration = iteration

    def set_dt(self, dt):
        '''Set time step'''
        self.dt = dt

    @check_class(phd.DomainManager)
    def set_domain_manager(self, domain_manager):
        '''Set domain manager for communiating across processors'''
        self.domain_manager = domain_manager

    @check_class(phd.EquationStateBase)
    def set_equation_state(self, equation_state):
        '''Set equation of state for gas'''
        self.equation_state = equation_state

    @check_class(phd.CarrayContainer)
    def set_particles(self, particles):
        '''Set particles to simulate'''
        self.particles = particles

    @check_class(phd.Mesh)
    def set_mesh(self, mesh):
        '''Set spatial mesh'''
        self.mesh = mesh

    @check_class(phd.ReconstructionBase)
    def set_reconstruction(self, reconstruction):
        '''Set reconstruction method'''
        self.reconstruction = reconstruction

    @check_class(phd.RiemannBase)
    def set_riemann(self, riemann):
        '''Set riemann solver'''
        self.riemann = riemann

class StaticMesh(IntegrateBase):
    '''
    Static mesh integrator. Once the mesh is created in `begin_loop` method
    the mesh will stay static throughout the simulation.
    '''
    def __init__(self, param_initial_time=0., param_final_time=1.0, param_dim=2):
        """Constructor for the Integrator
        """
        super(StaticMesh, self).__init__(param_initial_time, param_final_time, param_dim)

    def initialize(self):
        if not self.mesh or\
                not self.riemann or\
                not self.particles or\
                not self.domain_manager or\
                not self.reconstruction:
            raise RuntimeError("Not all setters defined in StaticMesh")

        # make sure proper dimension specified
        dim = len(self.particles.named_groups["position"])
        if dim != self.param_dim:
            raise RuntimeError(
                "Inconsistent dimension specified in particles %d and integrator %d)" %\
                        (dim, self.param_dim))

        # initialize classes
        self.domain_manager.initialize()
        self.mesh.initialize()

        self.reconstruction.set_fields_for_reconstruction(self.particles)
        self.reconstruction.initialize()

        self.riemann.set_feilds_for_flux(self.particles)
        self.riemann.initialize()

    def begin_loop(self, simulation):
        '''
        Build initial mesh, the mesh is only built once.
        '''
        # ignored if in serial 
        self.domain_manager.partition()

        # build mesh with ghost particles and
        # geometric quantities (volumes, faces, area, ...)
        self.mesh.build_geometry(self.particles)

        # relax mesh if needed TODO

        # compute density, velocity, pressure, ...
        self.equation_state.primitive_from_conserative(self.particles)

        # assign velocities to mesh cells and faces 
        self.mesh.assign_generator_velocities(self.particles)
        self.mesh.assign_face_velocities()

    def compute_time_step(self):
        '''
        Compute time step for current state of the simulation.
        Works in serial and parallel.
        '''
        self.loc_dt[0] = self.riemann.compute_time_step(
                self.particles, self.equation_state)

        # if in parallel find smallest dt across TODO
        return self.loc_dt[0]

    def evolve_timestep(self):
        '''
        Solve the compressible gas equations
        '''
        phdLogger.info('Static Mesh Integrator: Begining integration step...')

        # solve the riemann problem at each face
        phdLogger.info('Static Mesh Integrator: Starting reconstruction...')
        self.reconstruction.compute_states(self.particles, self.mesh,
                self.equation_state, self.riemann, self.domain_manager,
                self.dt, self.param_dim)
        phdLogger.success('Static Mesh Integrator - Finished reconstruction')

        phdLogger.info('Static Mesh Integrator: Starting riemann...')
        self.riemann.compute_fluxes(self.particles, self.mesh, self.reconstruction,
                self.equation_state, self.param_dim)
        phdLogger.success('Static Mesh Integrator: Finished riemann')

        self.mesh.update_from_fluxes(self.particles, self.riemann)

        # setup the mesh for the next setup 
        self.equation_state.primitive_from_conserative(self.particles)
        self.iteration += 1; self.time += self.dt

        phdLogger.info('Static Mesh Integrator: Finished integration')

    def after_loop(self):
        pass

#class MovingMesh(StaticMesh):
#    '''
#    Moving mesh integrator.
#    '''
#    def __init__(self, param_time=0, param_final_time=1.0, param_iteration=0):
#
#        # call super
#        self.param_time = param_time
#        self.param_iteration = param_iteration
#        self.param_final_time = param_final_time
#
#    def evolve_timestep(self):
#        '''
#        Solve the compressible gas equations
#        '''
#        phdLogger.info('Moving Mesh Integrator - Starting integration...')
#
#        # assign velocities to mesh cells and faces 
#        self.mesh.assign_generator_velocities(self.particles)
#        self.mesh.assign_face_velocities(self.particles)
#
#        # construct states at each face
#        phdLogger.info('Moving Mesh Integrator - Computing reconstruction...')
#        self.reconstruction.compute_states(self.particles, self.mesh,
#                self.equation_state, self.riemann, dt)
#        phdLogger.success('Moving Mesh Integrator - Finished reconstruction')
#
#        # solve riemann problem at each face and update variables
#        phdLogger.info('Moving Mesh Integrator - Solving riemann...')
#        self.riemann.compute_fluxes(self.particles, self.reconstruction)
#        self.mesh.update_from_fluxes(self.particles, self.riemann, self.dt)
#        phdLogger.success('Moving Mesh Integrator - Finished riemann')
#
#        # update mesh generator positions
#        self.domain_manager.move_generators(self.particles)
#        self.domain_manager.migrate_boundary_particles(self.particles)
#
#        # ignored if serial run
#        if self.domain_manager.check_for_partion():
#            phdLogger.info('Moving Mesh Integrator - Starting domain decomposition...')
#            self.domain_manager.partion()
#            phdLogger.success('Moving Mesh Integrator - Finished domain decomposition')
#
#        # setup the mesh for the next setup 
#        phdLogger.info('Moving Mesh Integrator - Rebuilding mesh...')
#        self.mesh.build_geometry(self.particles)
#        phdLogger.success('Moving Mesh Integrator - Finished mesh')
#
#        self.equation_state.primitive_from_conserative(self.particles)
#        self.equation_state.sound_speed(self.particles)
#        self.iteration += 1; self.time += self.dt
#
#        phdLogger.success('Moving Mesh Integrator - Finished integration')


## --------------------- to add after serial working ---------------------------
#class StaticMesh(IntegrateBase):
#    def begin_loop(self, simulation):
#        # relax mesh if needed
#        if self.mesh.param_relax_mesh and not simulation.param_restart:
#            for i in range(self.mesh.param_number_relaxations):
#
#                # lloyd relaxation
#                self.mesh.relax(self.particles)
#
#                # output data if requested
#                simulation.simulation_time(self)
#    def compute_time_step(self):
#
#        # if in parallel find smallest dt across
#        # all meshes 
#        if phd._in_parallel:
#            self.domain_manager.reduction(send=local_dt,
#                    rec=global_dt, op='min')
#            return self.glb_dt[0]
