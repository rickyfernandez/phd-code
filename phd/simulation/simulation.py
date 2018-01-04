import os
import phd
import logging

from ..integrate.integrate import IntegrateBase
from ..io.simulation_time import SimulationTime

from ..utils.logo import logo_str
from ..utils.tools import check_class, class_dict
from ..utils.logger import phdLogger, ufstring, original_emitter

class SimulationTAGS:
    """Tags to signal what state the simulation is in."""
    PRE_EVOLVE = 0
    BEFORE_LOOP = 1
    MAIN_LOOP = 2
    AFTER_LOOP = 3
    POST_EVOLVE = 4


class Simulation(object):
    """Marshalls the simulation.

    This class runs the simulation and all necessary outputs. It takes
    an integrator and simulation time and builds an output directory
    where files will be saved.

    Attributes
    ----------
    colored_logs : bool
        Output colored logs to screen if True otherwise revmove color.

    initial_timestep_factor : float
        For dt at the first iteration, reduce by this factor.

    integrator : IntegrateBase
        Advances the fluid equations by one step.

    log_level : str
        Level which logger is outputted.

    max_dt_change : float
        Largest change allowed of dt relative to old_dt.

    mesh_relax_iterations : int
        If non zero it signals the integrator to perform that
        many numbers of mesh relaxtion in before_loop.

    output_type : str
        Format which data is written to disk.

    simulation_name : str
       Name of problem solving, this name prefixs output data.

    simulation_time : SimulationTime
        Signals when to output data and finish the simulation.

    state : int
        Integer describing if simulation is before, in, or after
        main loop.

    """
    def __init__(
            self, max_dt_change=2.0, initial_timestep_factor=1.0,
            simulation_name='simulation', mesh_relax_iterations,
            colored_logs=True, log_level='debug', restart=False):
        """Constructor for simulation.

        Parameters:
        -----------
        colored_logs : bool
            Output colored logs to screen if True otherwise remove color.

        initial_timestep_factor : float
            For dt at the first iteration, reduce by this factor.

        log_level : str
            Level which logger is outputted.outputted.

        max_dt_change : float
            Largest change allowed of dt relative to old_dt.

        restart : bool
            Flag to signal if the simulation is a restart.

        simulation_name : str
           Name of problem solving, this name prefixs output data.

        """
        # integrator uses a setter 
        self.integrator = None
        self.simulation_time = None

        # state of the simulation
        self.restart = restart
        self.state = None

        # perform lloyd relaxtion scheme if non-zero
        self.mesh_relax_iterations = mesh_relax_iterations

        # time step parameters
        self.max_dt_change = max_dt_change
        self.initial_timestep_factor = initial_timestep_factor

        self.simulation_name = simulation_name
        self.output_directory = self.simulation_name + "_output"

        # create log file
        self.log_filename = self.simulation_name + ".log"
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setFormatter(logging.Formatter(ufstring))
        phdLogger.addHandler(file_handler)

        # set logger level for outputs
        if log_level == "debug":
            phdLogger.setLevel(logging.DEBUG)
        elif log_level == "info":
            phdLogger.setLevel(logging.INFO)
        elif log_level == "success":
            phdLogger.setLevel(logging.SUCCESS)
        elif log_level == "warning":
            phdLogger.setLevel(logging.WARNING)
        else:
            raise RuntimeError("Unknown log level: %s" % log_level)

        self.log_level = log_level

        # remove color output if desired
        if not colored_logs:
            sh = phdLogger.handlers[0]
            sh.setFormatter(logging.Formatter(ufstring))
            sh.emit = original_emitter
        self.colored_logs = colored_logs

        # create directory to store outputs
        if phd._rank == 0:
            if os.path.isdir(self.output_directory):
                phdLogger.warning("Directory %s already exists, "
                        "files maybe over written!" % self.output_directory)
            else:
                os.mkdir(self.output_directory)

    @check_class(IntegrateBase)
    def set_integrator(self, integrator):
        """Set integrator to evolve the simulation."""
        self.integrator = integrator

    @check_class(SimulationTime)
    def set_simulation_time(self, simulation_time):
        """Set time outputer for data outputs and ending the simulation"""
        self.simulation_time = simulation_time

    def solve(self):
        """Advance the simulation to final time.

        Main driver to evolve the equations. Responsible for advancing
        the simulation while outputting data to disk at appropriate
        times.

        """
        integrator = self.integrator
        simulation_time = self.simulation_time

        integrator.initialize()
        self.start_up_message()

        # output initial state of simulation
        self.state = SimulationTAGS.PRE_EVOLVE
        integrator.before_loop(self)

        # output initial data
        self.state = SimulationTAGS.BEFORE_LOOP
        simulation.simulation_time.output(
                self.output_directory,
                self)

        # evolve the simulation
        self.state = SimulationTAGS.MAIN_LOOP
        phdLogger.info("Beginning integration loop")
        while not simulation_time.finished(self):

            # compute new time step
            self.compute_time_step()

            # advance one time step
            integrator.evolve_timestep()

            # output if needed
            simulation_time.output(
                    self.output_directory,
                    self)

        # output final data
        self.state = SimulationTAGS.AFTER_LOOP
        simulation.simulation_time.output(
                self.output_directory,
                self)

        # clean up or last calculations
        self.state = SimulationTAGS.POST_EVOLVE
        integrator.after_loop(self)

        phdLogger.success("Simulation successfully finished!")

    def start_up_message(self):
        """Print out welcome message with details of the simulation."""
        bar = "-"*30
        message = "\n" + logo_str
        message += "\nSimulation Information\n" + bar

        if phd._in_parallel:
            message += "\nRunning in parallel: number of " +\
                "processors = %d" % phd._comm.Get_size()
                #"processors = %d" % self.num_procs
        else:
            message += "\nRunning in serial"

        # simulation name and output directory
        message += "\nLog file saved at: %s" % self.log_filename
        message += "\nProblem solving: %s" % self.simulation_name
        message += "\nOutput data will be saved at: %s\n" %\
                self.output_directory

        # print which classes are used in simulation
        cldict = class_dict(self.integrator)
        cldict["integrator"] = self.integrator.__class__.__name__
        message += "\nClasses used in the simulation\n" + bar + "\n"
        for key, val in sorted(cldict.iteritems()):
            message += key + ": " + val + "\n"

        phdLogger.info(message)

    def compute_time_step(self):
        """Modify time step for the next iteration.
        
        Calculate the time step is calculated from the integrator,
        then constrain by outputters and simulation.
        """
        # calculate new time step for integrator
        self.integrator.compute_time_step()

        # modify time step
        dt = self.integrator.dt
        if self.integrator.iteration == 0 and not self.restart:
            # shrink if first iteration
            dt = self.initial_timestep_factor*dt
        else:
            # constrain rate of change
            dt = min(self.max_dt_change*self.integrator.old_dt, dt)
        self.integrator.old_dt = dt

        # ensure the simulation outputs and finishes at selected time
        dt = min(self.simulation_time.modify_timestep(self), dt)
        self.integrator.set_dt(dt)
