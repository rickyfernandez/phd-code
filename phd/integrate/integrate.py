import logging
import numpy as np

import phd

from ..mesh.mesh import Mesh
from ..utils.tools import check_class
from ..domain.domain import DomainLimits
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

    domain_limits : DomainLimits
        Class that holds info of spatial extent.

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
        self.domain_limits = None
        self.reconstruction = None
        self.domain_manager = None
        self.boundary_condition = None

        if phd._in_parallel:
            # for communication dt across processors
            self.local_dt  = np.zeros(1, dtype=np.float64)
            self.global_dt = np.zeros(1, dtype=np.float64)

    @check_class(BoundaryConditionBase)
    def set_boundary_condition(self, boundary_condition):
        """Set domain manager for communiating across processors."""
        self.boundary_condition = boundary_condition

    @check_class(DomainLimits)
    def set_domain_limits(self, domain_limits):
        """Set domain manager for communiating across processors."""
        self.domain_limits = domain_limits

    @check_class(DomainManager)
    def set_domain_manager(self, domain_manager):
        """Set domain manager for communiating across processors."""
        self.domain_manager = domain_manager

    @check_class(EquationStateBase)
    def set_equation_state(self, equation_state):
        """Set equation of state for gas."""
        self.equation_state = equation_state

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
#    def add_source_term(self, source_term):
#        self.sources[source_term.__class__.__name__] = source_term

    def initialize(self):
        """Setup all connections for computation classes."""
        msg = "IntegrateBase::initialize called!"
        raise NotImplementedError(msg)

    def compute_time_step(self):
        """Compute time step for current state of the simulation."""
        msg = "IntegrateBase::compute_time_step called!"
        raise NotImplementedError(msg)

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
                self.mesh.relax(self.particles, self.domain_manager)

            # build mesh with ghost
            self.mesh.build_geometry(self.particles, self.domain_manager)

        # compute density, velocity, pressure, ...
        self.equation_state.conservative_from_primitive(self.particles)

        # should this be removed? TODO
        # assign cell and face velocities to zero 
        dim = len(self.particles.carray_named_groups["position"])
        for axis in "xyz"[:dim]:
            self.particles["w-" + axis][:] = 0.
            self.mesh.faces["velocity-" + axis][:] = 0.

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
            phd._comm.Allreduce(sendbuf=self.local_dt,
                    recvbuf=self.global_dt, op=phd.MPI.MIN)
            dt = self.global_dt[0]

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
                not self.domain_limits or\
                not self.domain_manager or\
                not self.equation_state or\
                not self.reconstruction or\
                not self.boundary_condition:
            raise RuntimeError("ERROR: Not all setters defined in %s!" %\
                    self.__class__.__name__)

        # initialize domain manager
        self.domain_manager.set_domain_limits(self.domain_limits)
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

    def evolve_timestep(self):
        """Solve the compressible gas equations."""

        phdLogger.info("StaticMeshMUSCLHancock: Starting integration")

        # build left/right states at each face in the mesh
        self.reconstruction.compute_gradients(self.particles, self.mesh,
                self.domain_manager)
        self.reconstruction.compute_states(self.particles, self.mesh,
                self.equation_state.get_gamma(), self.domain_manager, 0.5*self.dt,
                False)

        # solve riemann problem, generate flux
        self.riemann.compute_fluxes(self.particles, self.mesh, self.reconstruction,
                self.equation_state)

        # update conservative from fluxes
        self.mesh.update_from_fluxes(self.particles, self.riemann, self.dt)

        # convert updated conservative to primitive
        self.equation_state.primitive_from_conservative(self.particles)
        self.iteration += 1; self.time += self.dt

class MovingMeshMUSCLHancock(StaticMeshMUSCLHancock):
    """Moving mesh integrator."""
    def evolve_timestep(self):
        """Evolve the simulation for one time step."""

        phdLogger.info("MovingMeshMUSCLHancock: Starting integration")

        # assign velocities to mesh cells and faces 
        self.mesh.assign_generator_velocities(self.particles, self.equation_state)
        self.mesh.assign_face_velocities(self.particles)

        # build left/right states at each face in the mesh
        self.reconstruction.compute_gradients(self.particles, self.mesh,
                self.domain_manager)
        self.reconstruction.compute_states(self.particles, self.mesh,
                self.equation_state.get_gamma(), self.domain_manager, 0.5*self.dt,
                self.riemann.boost)

        # solve riemann problem, generate flux
        self.riemann.compute_fluxes(self.particles, self.mesh, self.reconstruction,
                self.equation_state)

        # update conservative from fluxes
        self.mesh.update_from_fluxes(self.particles, self.riemann, self.dt)

        # update mesh generator positions
        self.domain_manager.move_generators(self.particles, self.dt)

        # ignored if serial run
        if self.domain_manager.check_for_partition(self.particles, self):
            self.domain_manager.partion(self.particles)
        else:
            self.domain_manager.migrate_particles(self.particles)

        # setup the mesh for the next setup 
        self.mesh.build_geometry(self.particles, self.domain_manager)

        # convert updated conservative to primitive
        self.equation_state.primitive_from_conservative(self.particles)
        self.iteration += 1; self.time += self.dt
