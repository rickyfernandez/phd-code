import os
import phd
import logging

from ..integrate.integrate import IntegrateBase
from ..io.simulation_time import SimulationTime

from ..utils.logo import logo_str
from ..utils.tools import check_class, class_dict
from ..utils.logger import phdLogger, ufstring, original_emitter


class Simulation(object):
    """Marshalls the simulation

    This class runs the simulation and all necessary outputs. It takes
    an integrator and simulation time and builds an output directory
    where files will be saved.

    Attributes
    ----------
    integrator : IntegrateBase
        Advances the fluid equations by one step

    mesh_relax_iterations : int
        If non zero it signals the integrator to perform that
        many numbers of mesh relaxtion in before_loop

    param_max_dt_change : float
        Largest change allowed of dt relative to old_dt

    param_initial_timestep_factor : float
        For dt at the first iteration, reduce by this factor

    param_simulation_name : str
       Name of problem solving, this name prefixs output data.

    param_output_type : str
        Format which data is written to disk

    param_log_level : str
        Level which logger is outputted

    param_colored_logs : bool
        Output colored logs to screen if True otherwise revmove color

    simulation_time : SimulationTime
        Signals when to output data and finish the simulation
    """
    def __init__(
            self, max_dt_change=1.e33, initial_timestep_factor=1.0,
            simulation_name='simulation', colored_logs=True,
            log_level='debug'):
        """Constructor for simulation.

        Parameters:
        -----------
        max_dt_change : float
            Largest change allowed of dt relative to old_dt (param_max_dt_change*old_dt)

        initial_timestep_factor : float
            For dt at the first iteration, reduce by this factor

        simulation_name : str
           Name of problem solving, this name prefixs output data.

        log_level : str
            Level which logger is outputted

        colored_logs : bool
            Output colored logs to screen if True otherwise revmove color
        """
        # integrator uses a setter 
        self.integrator = None
        self.simulation_time = None

        self.mesh_relax_iterations = 0

        # time step parameters
        self.param_max_dt_change = max_dt_change
        self.param_initial_timestep_factor = initial_timestep_factor

        self.param_simulation_name = simulation_name
        self.param_output_directory = self.param_simulation_name + "_output"

        # create log file
        self.log_filename = self.param_simulation_name + ".log"
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setFormatter(logging.Formatter(ufstring))
        phdLogger.addHandler(file_handler)

        # set logger level
        if log_level == 'debug':
            phdLogger.setLevel(logging.DEBUG)
        elif log_level == 'info':
            phdLogger.setLevel(logging.INFO)
        elif log_level == 'success':
            phdLogger.setLevel(logging.SUCCESS)
        elif log_level == 'warning':
            phdLogger.setLevel(logging.WARNING)
        else:
            raise RuntimeError("Unknown log level: %s" % log_level)

        self.param_log_level = log_level

        # remove color output if desired
        if not colored_logs:
            sh = phdLogger.handlers[0]
            sh.setFormatter(logging.Formatter(ufstring))
            sh.emit = original_emitter
        self.param_colored_logs = colored_logs

        # create directory to store outputs
        if phd._comm.Get_rank() == 0:
            if os.path.isdir(self.param_output_directory):
                phdLogger.warning("Directory %s already exists, "
                        "files maybe over written!"  % self.param_output_directory)
            else:
                os.mkdir(self.param_output_directory)

    def set_mesh_relax_iterations(self, mesh_relax_iterations):
        self.mesh_relax_iterations = mesh_relax_iterations

    @check_class(IntegrateBase)
    def set_integrator(self, integrator):
        """
        Set integrator to evolve the simulation.
        """
        self.integrator = integrator

    @check_class(SimulationTime)
    def set_simulation_time(self, simulation_time):
        """
        Set time outputer for data outputs
        """
        self.simulation_time = simulation_time

    def solve(self):
        """
        Main driver to evolve the equations. Responsible for advancing
        the simulation while outputting data to disk at appropriate
        times.
        """
        integrator = self.integrator
        simulation_time = self.simulation_time

        integrator.initialize()
        self.start_up_message()

        # output initial state of simulation
        integrator.before_loop(self)

        # compute first time step
        integrator.compute_time_step()
        self.modify_timestep()

        # evolve the simulation
        phdLogger.info("Beginning integration loop")
        while not simulation_time.finished(integrator):

            # advance one time step
            integrator.evolve_timestep()

            # output if needed
            simulation_time.output(
                    self.param_output_directory,
                    integrator)

            # compute new time step
            integrator.compute_time_step()
            self.modify_timestep()

        # clean up or last calculations
        integrator.after_loop(self)
        phdLogger.success("Simulation successfully finished!")

    def start_up_message(self):
        '''
        Print out welcome message with details of the simulation
        '''
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
        message += "\nProblem solving: %s" % self.param_simulation_name
        message += "\nOutput data will be saved at: %s\n" %\
                self.param_output_directory

        # print which classes are used in simulation
        cldict = class_dict(self.integrator)
        cldict["integrator"] = self.integrator.__class__.__name__
        message += "\nClasses used in the simulation\n" + bar + "\n"
        for key, val in sorted(cldict.iteritems()):
            message += key + ": " + val + "\n"

        # log message
        phdLogger.info(message)

    def modify_timestep(self):
        '''
        Compute time step for the next iteration. First the integrator time step
        is called. Then it is modified by the simulation as to ensure it is
        constrained.
        '''
        dt = self.integrator.dt
        #if self.integrator.iteration == 0:
        #    # shrink if first iteration
        #    dt = self.param_initial_timestep_factor*dt
        #else:
        #    # constrain rate of change
        #    dt = min(self.param_max_dt_change*self.old_dt, dt)
        #self.old_dt = dt

        # ensure the simulation outputs and finishes at selected time
        dt = min(self.simulation_time.modify_timestep(self.integrator), dt)
        self.integrator.set_dt(dt)
