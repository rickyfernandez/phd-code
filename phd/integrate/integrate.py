import logging
import numpy as np

import phd

from ..mesh.mesh import Mesh
from ..utils.tools import check_class
from ..riemann.riemann import RiemannBase
from ..domain.domain_manager import DomainManager
from ..containers.containers import CarrayContainer
from ..domain.boundary import BoundaryConditionBase
from ..equation_state.equation_state import EquationStateBase
from ..reconstruction.reconstruction import ReconstructionBase

phdLogger = logging.getLogger("phd")

class IntegrateBase(object):
    """Class that solves the fluid equations.

    Every integration implementation has to inherit this class and follow
    the api.

    Attributes
    ----------
    boundary_condition : BoundaryConditionBase
        Class that creates ghost particles from given boundary
        condition.

    domain_manager : DomainManager
        Class that handels all things related with the domain.

    dt : float
        Time step of the simulation.

    initial_timestep_factor : float
        For dt at the first iteration, reduce by this factor.

    max_dt_change : float
        Largest change allowed of dt relative to old_dt.

    mesh : Mesh
        Class that builds the domain mesh.

    old_dt : float
        Previous time step.

    particles : CarrayContainer
        Class that holds all information pertaining to the particles
        in the simulation.

    reconstruction : ReconstructionBase
        Class that performs field reconstruction inputs for the
        riemann problem.

    riemann : RiemannBase
        Class that solves the riemann problem.

    time : float
        Current time of the simulation.

    iteration : int
        Current iteration of the simulation.

    """
    def __init__(self, dt=0., time=0., iteration=0, old_dt=-np.inf,
                 max_dt_change=2.0, initial_timestep_factor=1.0,
                 restart=False, **kwargs):
        """Constructor for Integrate base class.

        Parameters
        ----------
        dt : float
            Time step of the simulation.

        initial_timestep_factor : float
            For dt at the first iteration, reduce by this factor.

        max_dt_change : float
            Largest change allowed of dt relative to old_dt.

        old_dt : float
            Previous time step.

        time : float
            Current time of the simulation.

        iteration : int
            Current iteration of the simulation.

        restart : bool
            Flag to signal if the simulation is a restart.

        """
        self.dt = dt
        self.time = time
        self.old_dt = old_dt
        self.iteration = iteration

        self.restart = restart

        # time step parameters
        self.max_dt_change = max_dt_change
        self.initial_timestep_factor = initial_timestep_factor

        # required objects to be set
        self.mesh = None
        self.riemann = None
        self.particles = None
        self.reconstruction = None
        self.domain_manager = None
        self.boundary_condition = None

        self.source_terms = {}

        if phd._in_parallel:

            self.load_balance = None

            # for communication dt across processors
            self.local_dt  = np.zeros(1, dtype=np.float64)
            self.global_dt = np.zeros(1, dtype=np.float64)

    @check_class(BoundaryConditionBase)
    def set_boundary_condition(self, boundary_condition):
        """Set domain manager for communiating across processors."""
        self.boundary_condition = boundary_condition

    @check_class(DomainManager)
    def set_domain_manager(self, domain_manager):
        """Set domain manager for communiating across processors."""
        self.domain_manager = domain_manager

    @check_class(EquationStateBase)
    def set_equation_state(self, equation_state):
        """Set equation of state for gas."""
        self.equation_state = equation_state

    #@check_class(EquationStateBase)
    def set_load_balance(self, load_balance):
        """Set equation of state for gas."""
        self.load_balance = load_balance

    @check_class(CarrayContainer)
    def set_particles(self, particles):
        """Set particles to simulate."""
        self.particles = particles

    @check_class(Mesh)
    def set_mesh(self, mesh):
        """Set spatial mesh."""
        self.mesh = mesh

    @check_class(ReconstructionBase)
    def set_reconstruction(self, reconstruction):
        """Set reconstruction method."""
        self.reconstruction = reconstruction

    @check_class(RiemannBase)
    def set_riemann(self, riemann):
        """Set riemann solver."""
        self.riemann = riemann

#    @check_class(SourceTermBase)
    def add_source_term(self, source_term):
        self.source_terms[source_term.__class__.__name__] = source_term

    def initialize(self):
        """Setup all connections for computation classes."""
        msg = "IntegrateBase::initialize called!"
        raise NotImplementedError(msg)

    def compute_source(self, term):
        if self.source_terms:
            for source in self.source_terms.itervalues():
                if term == "motion":
                    source.apply_motion(self)
                elif term == "primitive":
                    source.apply_primitive(self)
                elif term == "conservative":
                    source.apply_conservative(self)
                elif term == "flux":
                    source.apply_flux(self)
                elif term == "compute":
                    source.compute_source(self)
                else:
                    raise RuntimeError("ERROR: Unknown source term method")

    def before_loop(self, simulation):
        """Perform any operations before the main loop of the simulation.

        Any calculations or setups should be performed in this call. When
        creating a new class ovewrite this method but be sure to call this
        base method or copy the parts to needed. This base implementation
        performs the first domain split if in parallel and mesh relaxation.

        Parameters
        ----------
        simulation : Simulation
           Class that marshalls the simulation of the fluid equations.

        """
        # ignored if in serial 
        self.domain_manager.partition(self.particles)
        phdLogger.info("IntegrateBase: Building initial mesh")

        # build mesh with ghost particles and
        # geometric quantities (volumes, faces, area, ...)
        self.mesh.build_geometry(self.particles, self.domain_manager)

        # relax mesh if needed 
        if self.mesh.relax_iterations > 0 and not self.restart:
            phdLogger.info("Relaxing mesh:")

            for i in range(self.mesh.relax_iterations):
                phdLogger.info("Relaxing iteration %d" % i)
                simulation.simulation_time_manager.output(simulation)
                self.domain_manager.partition(self.particles)
                self.mesh.relax(self.particles, self.domain_manager)

            # build mesh with ghost
            self.domain_manager.partition(self.particles)
            self.mesh.build_geometry(self.particles, self.domain_manager)

        self.domain_manager.boundary_condition.update_fields(
                self.particles, self.domain_manager)

        # compute mass, momentum ...
        self.equation_state.conservative_from_primitive(self.particles)

        # compute source terms
        self.compute_source("compute")

    def compute_time_step(self):
        """Compute time step for current state of simulation.

        Calculate the time step is then constrain by outputters
        and simulation.
        """
        # calculate new time step for integrator
        dt = self.riemann.compute_time_step(
                self.particles, self.equation_state)

        if phd._in_parallel:

            self.local_dt[0] = dt
            phd._comm.Allreduce(
                    [self.local_dt,  phd.MPI.DOUBLE],
                    [self.global_dt, phd.MPI.DOUBLE],
                    op=phd.MPI.MIN)
            dt = self.global_dt[0]

        self.dt = dt
        phdLogger.info("Hydro dt: %f" %dt)

        # modify timestep from source terms
        if self.source_terms:
            dt_source = np.inf
            for source in self.source_terms.itervalues():
                dt_source = min(source.compute_time_step(self), dt_source)

            if phd._in_parallel:

                self.local_dt[0] = dt_source
                phd._comm.Allreduce(
                        [self.local_dt,  phd.MPI.DOUBLE],
                        [self.global_dt, phd.MPI.DOUBLE],
                        op=phd.MPI.MIN)
                dt_source = self.global_dt[0]

            phdLogger.info("Source dt: %f" %dt_source)
            dt = min(dt, dt_source)

        # modify time step
#        if self.iteration == 0 and not self.restart:
#            # shrink if first iteration
#            dt = self.initial_timestep_factor*dt
#        else:
#            # constrain rate of change
#            dt = min(self.max_dt_change*self.old_dt, dt)
#            self.old_dt = self.dt

        self.dt = dt

        return dt

    def evolve_timestep(self):
        """Evolve the simulation for one time step."""
        msg = "IntegrateBase::evolve_timestep called!"
        raise NotImplementedError(msg)

    def after_loop(self, simulation):
        pass


class StaticMeshMUSCLHancock(IntegrateBase):
    """Static mesh integrator.

    Once the mesh is created in `begin_loop` method
    the mesh will stay static throughout the simulation.

    """
    def initialize(self):
        if not self.mesh or\
                not self.riemann or\
                not self.particles or\
                not self.domain_manager or\
                not self.equation_state or\
                not self.reconstruction or\
                not self.boundary_condition:
            raise RuntimeError("ERROR: Not all setters defined in %s!" %\
                    self.__class__.__name__)

        if phd._in_parallel:
            if not self.load_balance:
                raise RuntimeError("ERROR: Load Balance setter not defined")

            self.load_balance.add_domain_info(self.domain_manager)
            self.load_balance.initialize()

            self.domain_manager.set_load_balance(
                    self.load_balance)

        # initialize domain manager
        self.domain_manager.set_boundary_condition(self.boundary_condition)
        self.domain_manager.register_fields(self.particles)
        self.domain_manager.initialize()

        # intialize mesh
        self.mesh.register_fields(self.particles)
        self.mesh.initialize()

        # intialize reconstruction
        self.reconstruction.add_fields(self.particles)
        self.reconstruction.initialize()

        # intialize riemann
        self.riemann.add_fields(self.particles)
        self.riemann.initialize()

    def before_loop(self, simulation):
        super(StaticMeshMUSCLHancock, self).before_loop(simulation)

        # zero out mesh moition for static
        dim = len(self.particles.carray_named_groups["position"])
        for i in "xyz"[:dim]:
            self.particles["w-"+i][:] = 0.
            self.mesh.faces["velocity-"+i][:] = 0.

    def evolve_timestep(self):
        """Solve the compressible gas equations."""

        phdLogger.info("StaticMeshMUSCLHancock: Starting integration")

        # build left/right states at each face in the mesh
        self.reconstruction.compute_gradients(self.particles, self.mesh,
                self.domain_manager)
        self.reconstruction.compute_states(self.particles, self.mesh,
                self.equation_state.get_gamma(), self.domain_manager, 0.5*self.dt,
                self.riemann.boost)
        self.compute_source("primitive")

        # solve riemann problem, generate flux
        self.riemann.compute_fluxes(self.particles, self.mesh, self.reconstruction,
                self.equation_state)

        # update conservative from fluxes
        self.mesh.update_from_fluxes(self.particles, self.riemann, self.dt)
        self.compute_source("flux")
        self.compute_source("compute")

        self.compute_source("conservative")
        self.domain_manager.update_ghost_fields(self.particles,
                self.particles.carray_named_groups["conservative"],
                True)

        # convert updated conservative to primitive
        self.equation_state.primitive_from_conservative(self.particles)
        self.iteration += 1; self.time += self.dt

class MovingMeshMUSCLHancock(StaticMeshMUSCLHancock):
    """Moving mesh integrator."""
    def before_loop(self, simulation):
        super(MovingMeshMUSCLHancock, self).before_loop(simulation)

    def evolve_timestep(self):
        """Evolve the simulation for one time step."""

        phdLogger.info("MovingMeshMUSCLHancock: Starting integration")

        # assign velocities to mesh cells and faces 
        self.mesh.assign_generator_velocities(self.particles, self.equation_state)
        self.compute_source("motion")
        self.mesh.assign_face_velocities(self.particles)

        # build left/right states at each face in the mesh
        self.reconstruction.compute_gradients(self.particles, self.mesh,
                self.domain_manager)
        self.reconstruction.compute_states(self.particles, self.mesh,
                self.equation_state.get_gamma(), self.domain_manager, 0.5*self.dt,
                self.riemann.boost)
        self.compute_source("primitive")

        # solve riemann problem, generate flux
        self.riemann.compute_fluxes(self.particles, self.mesh, self.reconstruction,
                self.equation_state)

        # update conservative from fluxes
        self.mesh.update_from_fluxes(self.particles, self.riemann, self.dt)
        self.compute_source("flux")

        # update mesh generator positions
        self.domain_manager.move_generators(self.particles, self.dt)

        # for moved particles apply boundary conditions and
        # move particles to correct processor
        self.domain_manager.partition(self.particles)

        # setup the mesh with ghost particles for the next setup 
        self.mesh.build_geometry(self.particles, self.domain_manager)
        self.domain_manager.boundary_condition.update_fields(
                self.particles, self.domain_manager)

        self.compute_source("compute")

        # convert updated conservative to primitive
        self.compute_source("conservative")
        self.equation_state.primitive_from_conservative(self.particles)
        self.iteration += 1; self.time += self.dt


class Nbody(IntegrateBase):
    """Class that solves nbody problem.

    Nbody solver for collisionless particles without 
    hydrodynamics.

    Attributes
    ----------
    domain_manager : DomainManager
        Class that handels all things related with the domain.

    dt : float
        Time step of the simulation.

    particles : CarrayContainer
        Class that holds all information pertaining to the particles
        in the simulation.

    time : float
        Current time of the simulation.

    iteration : int
        Current iteration of the simulation.

    """
    def __init__(self, dt=0., time=0., iteration=0, restart=False, **kwargs):
        """Constructor for Nbody integrator.

        Parameters
        ----------
        dt : float
            Time step of the simulation.

        time : float
            Current time of the simulation.

        iteration : int
            Current iteration of the simulation.

        restart : bool
            Flag to signal if the simulation is a restart.

        """
        super(Nbody, self).__init__(dt=dt, time=time, iteration=iteration,
                restart=restart)

    def set_gravity_tree(self, gravity_tree):
        """Set equation of state for gas."""
        self.gravity_tree = gravity_tree

    def initialize(self):
        """Setup all connections for computation classes."""

        if not self.particles or\
                not self.domain_manager or\
                not self.gravity_tree:
            raise RuntimeError("ERROR: Not all setters defined in %s!" %\
                    self.__class__.__name__)

        if phd._in_parallel:
            if not self.load_balance:
                raise RuntimeError("ERROR: Load Balance setter not defined")

            self.load_balance.add_domain_info(self.domain_manager)
            self.load_balance.initialize()

            self.domain_manager.set_load_balance(
                    self.load_balance)

        # initialize domain manager
        self.domain_manager.register_fields(self.particles)
        # hack, we don't initialize, create Null Boundary condition
        #self.domain_manager.initialize()

        self.gravity_tree.add_fields(self.particles)
        self.gravity_tree.register_fields(self.particles)
        self.gravity_tree.set_domain_manager(self.domain_manager)
        self.gravity_tree.initialize()

        dim = len(self.particles.carray_named_groups["position"])
        self.axis = "xyz"[:dim]

    def before_loop(self, simulation):
        """Perform any operations before the main loop of the simulation.

        Parameters
        ----------
        simulation : Simulation
           Class that marshalls the simulation of the fluid equations.

        """
        if phd._in_parallel:
            self.load_balance.decomposition(self.particles)

        # calculate initial accelerations
        self.gravity_tree._build_tree(self.particles)
        self.gravity_tree.walk(self.particles)

        # output if needed
        simulation.simulation_time_manager.output(simulation)

    def compute_time_step(self):
        """Compute time step for current state of simulation.

        Calculate the time step is then constrain by outputters
        and simulation.
        """
        return self.dt

    def evolve_timestep(self):
        """Evolve the simulation for one time step."""

        # kick
        for ax in self.axis:
            self.particles["velocity-"+ax][:] += 0.5*self.dt*self.particles["acceleration-"+ax][:]

        # drift
        for ax in self.axis:
            self.particles["position-"+ax][:] += self.dt*self.particles["velocity-"+ax][:]

        if phd._in_parallel:
            self.load_balance.decomposition(self.particles)

        # compute new acceleration
        self.gravity_tree._build_tree(self.particles)
        self.gravity_tree.walk(self.particles)

        # kick
        for ax in self.axis:
            self.particles["velocity-"+ax][:] += 0.5*self.dt*self.particles["acceleration-"+ax][:]

        self.iteration += 1; self.time += self.dt

    def after_loop(self, simulation):
        pass

#class MovingMeshPakmor(StaticMeshMUSCLHancock):
#    """Moving mesh integrator."""
#    def first_stage(self, dt):
#        # assign velocities to mesh cells and faces 
#        self.mesh.assign_generator_velocities(self.particles, self.equation_state)
#        self.mesh.assign_face_velocities(self.particles)
#
#        # reconstruct with temporal component
#        self.reconstruction.compute_gradients(self.particles, self.mesh,
#                self.domain_manager)
#        self.reconstruction.compute_states(self.particles, self.mesh,
#                self.equation_state.get_gamma(), self.domain_manager, 0,
#                self.riemann.boost, False)
#
#        # solve riemann problem, generate flux
#        self.riemann.compute_fluxes(self.particles, self.mesh, self.reconstruction,
#                self.equation_state)
#
#        # update conservative from fluxes
#        self.mesh.update_from_fluxes(self.particles, self.riemann, 0.5*dt)
#
#    def second_stage(self, dt):
#        # communicate old gradients and boundary conditions
#        #self.reconstruction.grad.resize(self.particles.get_carray_size())
#
#        self.domain_manager.boundary_condition.update_fields(
#                self.particles, self.domain_manager)
#        #self.domain_manager.update_ghost_gradients(
#        #        self.particles, self.reconstruction.grad)
#
#        # assign velocities to mesh cells and faces 
#        self.mesh.assign_generator_velocities(self.particles, self.equation_state)
#        self.mesh.assign_face_velocities(self.particles)
#
#        self.reconstruction.compute_states(self.particles, self.mesh,
#                self.equation_state.get_gamma(), self.domain_manager, dt,
#                self.riemann.boost, True)
#
#        # solve riemann problem, generate flux
#        self.riemann.compute_fluxes(self.particles, self.mesh, self.reconstruction,
#                self.equation_state)
#
#        # update conservative from fluxes
#        self.mesh.update_from_fluxes(self.particles, self.riemann, 0.5*dt)
#
#        # transfer accumulated updates to ghost for 
#        # next time step
#        self.domain_manager.update_ghost_fields(
#                self.particles,
#                self.particles.carray_named_groups["conservative"],
#                True)
#
#        # convert updated conservative to primitive
#        self.equation_state.primitive_from_conservative(self.particles)
#
#    def evolve_timestep(self):
#        """Evolve the simulation for one time step."""
#
#        phdLogger.info("MovingMeshPakmor: Starting integration")
#
#        # add 1/2 update with current mesh
#        self.first_stage(self.dt)
#
#        # update mesh generator positions, apply boundary conditions
#        self.domain_manager.move_generators(self.particles, self.dt)
#        self.domain_manager.partition(self.particles)
#        self.mesh.build_geometry(self.particles, self.domain_manager)
#
#        # add 1/2 update with updated mesh
#        self.second_stage(self.dt)
#        self.iteration += 1; self.time += self.dt
