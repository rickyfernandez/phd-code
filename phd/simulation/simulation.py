import os
import phd
import logging

from ..integrate.integrate import IntegrateBase
from ..io.simulation_time import SimulationTime

from ..utils.logo import logo_str
from ..utils.tools import check_class, class_dict
from ..utils.logger import phdLogger, ufstring, original_emitter


class Simulation(object):
    """Marshalls the simulation."""
    def __init__(
        self, param_max_dt_change=1.e33, param_initial_timestep_factor=1.0,
        param_simulation_name='simulation', param_colored_logs=True, param_log_level='debug'):
        """Constructor for simulation.

        Parameters:
        -----------
        param_max_dt_change : float
            Largest change allowed of dt relative to old_dt (max_dt_change*old_dt)

        param_initial_timestep_factor : float
            For dt at the first iteration, reduce by this factor

        param_simulation_name : str
           Name of problem solving, this name prefixs output data.

        param_output_relax : bool
            Write out data at each mesh relaxation

        param_output_type : str
            Format which data is written to disk

        param_log_level : str
            Level which logger is outputted

        param_colored_logs : bool
            Output colored logs to screen if True otherwise revmove color
        """
        # integrator uses a setter 
        self.integrator = None
        self.simulation_time = None

        self.mesh_relax_iterations = 0

        # time step parameters
        self.param_max_dt_change = param_max_dt_change
        self.param_initial_timestep_factor = param_initial_timestep_factor

        # parallel parameters
        self.rank = 0
        self.comm = None
        self.num_procs = 1

        # if mpi4py is available
        if phd._in_parallel:
            self.comm = phd._comm
            self.num_procs = self.comm.Get_size()
            self.rank = self.comm.Get_rank()

        self.param_simulation_name = param_simulation_name
        self.param_output_directory = self.param_simulation_name + "_output"

        # create log file
        self.log_filename = self.param_simulation_name + ".log"
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setFormatter(logging.Formatter(ufstring))
        phdLogger.addHandler(file_handler)

        # set logger level
        if param_log_level == 'debug':
            phdLogger.setLevel(logging.DEBUG)
        elif param_log_level == 'info':
            phdLogger.setLevel(logging.INFO)
        elif param_log_level == 'success':
            phdLogger.setLevel(logging.SUCCESS)
        elif param_log_level == 'warning':
            phdLogger.setLevel(logging.WARNING)
        self.param_log_level = param_log_level

        # remove color output if desired
        if not param_colored_logs:
            sh = phdLogger.handlers[0]
            sh.setFormatter(logging.Formatter(ufstring))
            sh.emit = original_emitter
        self.param_colored_logs = param_colored_logs

        # create directory to store outputs
        if self.rank == 0:
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
                "processors = %d" % self.num_procs
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
