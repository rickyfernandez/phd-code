import os
import phd
import logging

from ..utils.logo import logo_str
from ..integrate.integrate import IntegrateBase
from ..utils.tools import check_class, class_dict
from ..io.simulation_time_manager import SimulationTimeManager
from ..utils.logger import phdLogger, ufstring, original_emitter
from ..utils.particle_tags import SimulationTAGS

class Simulation(object):
    """Marshalls the simulation.

    This class runs the simulation and all necessary outputs. It takes
    an integrator and simulation time and builds an output directory
    where files will be saved.

    Attributes
    ----------
    colored_logs : bool
        Output colored logs to screen if True otherwise remove color.

    integrator : IntegrateBase
        Advances the fluid equations by one step.

    log_level : str
        Level which logger is outputted.

    _output_directory : str
        Directory to store all output of the simulation.

    simulation_name : str
       Name of problem solving, this name prefixs output data.

    simulation_time_manager : SimulationTimeManager
        Signals when to output data and finish the simulation.

    _state : int
        Integer describing if simulation is before, in, or after
        main loop.

    """
    def __init__(self, simulation_name='simulation', colored_logs=True,
                 log_level='debug', **kwargs):
        """Constructor for simulation.

        Parameters:
        -----------
        colored_logs : bool
            Output colored logs to screen if True otherwise remove color.

        log_level : str
            Level which logger is outputted.outputted.

        simulation_name : str
           Name of problem solving, this name prefixs output data.

        """
        # variables use setters 
        self.integrator = None
        self.simulation_time_manager = None

        self._state = None

        self.log_level = log_level
        self.colored_logs = colored_logs

        self.simulation_name = simulation_name
        self._output_directory = self.simulation_name + "_output"

        # all log output stored in this file
        self.log_filename = self.simulation_name + ".log"

    def initialize(self):

        if not self.integrator or not self.simulation_time_manager:
            raise RuntimeError("Not all setters defined in Simulation!")

        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setFormatter(logging.Formatter(ufstring))
        phdLogger.addHandler(file_handler)

        # set logger level for outputs
        if self.log_level == "debug":
            phdLogger.setLevel(logging.DEBUG)
        elif self.log_level == "info":
            phdLogger.setLevel(logging.INFO)
        elif self.log_level == "success":
            phdLogger.setLevel(logging.SUCCESS)
        elif self.log_level == "warning":
            phdLogger.setLevel(logging.WARNING)
        else:
            raise RuntimeError("Unknown log level: %s" % self.log_level)

        # remove color output if desired
        if not self.colored_logs:
            sh = phdLogger.handlers[0]
            sh.setFormatter(logging.Formatter(ufstring))
            sh.emit = original_emitter

        # create directory to store outputs
        if phd._rank == 0:
            if os.path.isdir(self._output_directory):
                phdLogger.warning("Directory %s already exists, "
                        "files maybe over written!" % self._output_directory)
            else:
                os.mkdir(self._output_directory)

        # initialize all classes, riemann, reconstruction, ...
        self.integrator.initialize()

        # initialize all outputters
        for output in self.simulation_time_manager._outputs:
            output.set_output_directory(self._output_directory)
            output.initialize()

    @check_class(IntegrateBase)
    def set_integrator(self, integrator):
        """Set integrator to evolve the simulation."""
        self.integrator = integrator

    @check_class(SimulationTimeManager)
    def set_simulation_time_manager(self, simulation_time_manager):
        """Set time outputter for data outputs and ending the simulation"""
        self.simulation_time_manager = simulation_time_manager

    def solve(self):
        """Advance the simulation to final time.

        Main driver to evolve the equations. Responsible for advancing
        the simulation while outputting data to disk at appropriate
        times.

        """
        self.start_up_message()

        # perform any initial computation 
        self._state = SimulationTAGS.PRE_EVOLVE
        self.integrator.before_loop(self)

        # output initial data
        self._state = SimulationTAGS.BEFORE_LOOP
        self.simulation_time_manager.output(self)

        # evolve the simulation
        self._state = SimulationTAGS.MAIN_LOOP
        phdLogger.info("Beginning integration loop")
        while not self.simulation_time_manager.finished(self):

            # compute new time step
            self.compute_time_step()
            phdLogger.info("Starting iteration: %d time: %f dt: %f" %\
                    (self.integrator.iteration,
                     self.integrator.time,
                     self.integrator.dt))

            # advance one time step
            self.integrator.evolve_timestep()

            # output if needed
            self.simulation_time_manager.output(self)

        # output final data
        self._state = SimulationTAGS.AFTER_LOOP
        self.simulation_time_manager.output(self)

        # clean up or last calculations
        self._state = SimulationTAGS.POST_EVOLVE
        self.integrator.after_loop(self)

        phdLogger.success("Simulation successfully finished!")

    def start_up_message(self):
        """Print out welcome message with details of the simulation."""
        bar = "-"*30
        message = "\n" + logo_str
        message += "\nSimulation Information\n" + bar

        if phd._in_parallel:
            message += "\nRunning in parallel: number of " +\
                "processors = %d" % phd._size
        else:
            message += "\nRunning in serial"

        # simulation name and output directory
        message += "\nLog file saved at: %s" % self.log_filename
        message += "\nProblem solving: %s" % self.simulation_name
        message += "\nOutput data will be saved at: %s\n" %\
                self._output_directory

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
        # cfl constrained time-step
        dt = self.integrator.compute_time_step()

        # ensure the simulation outputs and finishes at selected time
        dt = min(self.simulation_time_manager.modify_timestep(self), dt)
        self.integrator.dt = dt
