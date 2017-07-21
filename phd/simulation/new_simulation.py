import os
import phd
import logging

from ..utils.logo import logo_str
from ..utils.tools import check_class, class_dict
from ..utils.logger import phdLogger, ufstring, original_emitter


class NewSimulation(object):
    """Marshalls the simulation."""
    def __init__(
        self, final_time=1.0, max_dt_change=1.e33, initial_timestep_factor=1.0, cfl=0.5,
        output_time_interval=100000, simulation_name='simulation', output_type='hdf5',
        colored_logs=True):
        """Constructor for simulation.

        Parameters:
        -----------
        final_time : float
            Final time of simulation

        max_dt_change : float
            Largest change allowed of dt relative to old_dt (max_dt_change*old_dt)

        initial_timestep_factor : float
            For dt at the first iteration, reduce by this factor

        clf : float
            Courant Friedrichs Lewy (CFL) condition

        output_time_interval : int
            Output at requested time interval. This will not output exatcly
            at the interval but once a new time multiple is reached.

        simulation_name : str
           Name of problem solving, this name prefixs output data.

        relax_num_iterations : int
            Number of relaxations performed on the mesh before evolving
            the equations.

        output_relax : bool
            Write out data at each mesh relaxation

        output_type : str
            Format which data is written to disk

        log_level : str
            Level which logger is outputted
        """
        # integrator uses a setter 
        self.integrator = None
        self.simulation_time = None

        # time step parameters
        self.cfl = cfl
        self.final_time = final_time
        self.max_dt_change = max_dt_change
        self.initial_timestep_factor = initial_timestep_factor

        # output parameters
        self.output_time_interval = output_time_interval

        # create direcotry to store outputs
        self.simulation_name = simulation_name
        self.output_directory = self.simulation_name + "_output"

        # parallel parameters
        self.rank = 0
        self.comm = None
        self.num_procs = 1

        # if mpi4py is available
        if phd._has_mpi:
            self.comm = phd._comm
            self.num_procs = self.comm.Get_size()
            self.rank = self.comm.Get_rank()

        # create log file
        self.log_filename = self.simulation_name + ".log"
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setFormatter(logging.Formatter(ufstring))
        phdLogger.addHandler(file_handler)

        if not colored_logs:
            sh = phdLogger.handlers[0]
            sh.setFormatter(logging.Formatter(ufstring))
            sh.emit = original_emitter

#        # create input/output type
#        if inout_type == 'hdf5':
#            self.input_output = Hdf5()
#        else:
#            RuntimeError('Output format not recognized: %s' % output_type)

    #@check_class(phd.IntegrateBase)
    def set_integrator(self, integrator):
        """
        Set integrator to evolve the simulation.
        """
        self.integrator = integrator

    def set_simulationtime(self, simulation_time):
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
        integrate = self.integrate
        simulation_time = self.simulation_time

        integrate.initialize()
        self.start_up_message()

        # output initial state of simulation
        integrate.before_loop(self)
        phdLogger.info("Writting initial output...")
        self.output_data()

        # evolve the simulation
        phdLogger.info("Beginning integration loop")
        while not simulation_time.finish(integration):

            phdLogger.info("Starting iteration: "
                    "%d time: %f dt: %f" %\
                    (integrator.iteration,
                     integrator.time,
                     integrator.dt))

            # advance one time step
            integrator.evolve_timestep()
            phdLogger.success("Finished iteration: "
                    "%d time: %f dt: %f" %\
                    integrator.iteration)

            # compute new time step
            integrator.compute_timestep()
            self.modify_timestep() # if needed

            # output if needed
            if simulation_time.outputs(integrator):
                phdLogger.info(
                        "Writting output at time %g, "
                        "iteration %d, dt %g: %s" %\
                        (integrator.iteration,
                         integrator.time,
                         integrator.dt))
                self.ouput_data()

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

        # print if serial or parallel run
        if phd._in_parallel:
            message += "\nRunning in parallel: number of " +\
                "processors = %d" % self.num_procs
        else:
            message += "\nRunning in serial"

        # simulation name and output directory
        message += "\nProblem solving: %s" % self.simulation_name
        message += "\nOutput data will be saved at: %s\n" %\
                self.output_directory

        # print which classes are used in simulation
        cldict = class_dict(self.integrator)
        message += "\nClasses used in the simulation\n" + bar + "\n"
        for key, val in cldict.iteritems():
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
        if self.integrator.iteration == 0:
            # shrink if first iteration
            dt = self.initial_timestep_factor*dt
        else:
            # constrain rate of change
            dt = min(max_dt_change*self.old_dt, dt)
        self.old_dt = dt

        # ensure the simulation stops at final time
        if self.integrator.time + dt > self.time_outputer.final_time:
            dt = self.final_time - self.integrator.time
        self.integrator.set_dt(dt)

#    def output_data(self):
#        """
#        """
#        output_dir = self.path + "/" + self.fname + "_" + `self.output`.zfill(4)
#
#        if self.parallel_run:
#            output_dir = output_dir + "/" + "data" + `self.output`.zfill(4)
#                + '_cpu' + `self.rank`.zfill(4)
#
#            if self.rank == 0:
#                os.mkdir(output_dir)
#            self.barrier()
#
#        f = h5py.File(ouput_name + '.hdf5', 'w')
#        for prop in self.pc.properties.keys():
#            f["/" + prop] = self.pc[prop]
#
#        f.attrs["iteration_count"] = iteration_count
#        f.attrs["time"] = current_time
#        f.attrs["dt"] = dt
#        f.close()
#
#        self.output += 1
#        self._barrier()

    def barrier(self):
        if phd._in_parallel:
            self.comm.barrier()
