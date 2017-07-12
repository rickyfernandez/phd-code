import os
import phd

from ..utils.logo import logo_str
from ..utils.logger import phdLogger
#from ..utils.tools import check_class

try:
    import mpi4py.MPI as mpi
    _found_mpi = True
except ImportError:
    _found_mpi = False


class Simulation(object):
    """Marshalls the simulation."""
    def __init__(
        self, final_time=1.0, max_dt_change=1.e33, initial_timestep_factor=1.0, cfl=0.5,
        simulation_name='simulation', output_type='hdf5'):
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

        # time step parameters
        self.cfl = cfl
        self.final_time = final_time
        self.max_dt_change = max_dt_change
        self.initial_timestep_factor = initial_timestep_factor

        # output parameters
        self.output_time_interval = output_time_interval

        # create direcotry to store outputs
        self.simulation_name = simulation_name
        self.output_directory = self.simulation_name + '_output'

        # parallel parameters
        self.rank = 0
        self.comm = None
        self.num_procs = 1
        self.parallel_run = True if _found_mpi else False

        # if mpi4py is available
        if self.parallel_run:
            self.comm = mpi.COMM_WORLD
            self.num_procs = comm.Get_size()
            self.rank = comm.Get_rank()

        # create log file
        self.log_filename = self.simulation_name + '.log'
        file_handler = logging.FileHandler(self.log_filename)
        formatter = logging.Formatter(ufstring)
        file_handler.setFormatter(formatter)
        phdLogger.addHandler(file_handler)

        if output_type == 'hdf5':
            self.output = Hdf5()
        else:
            RuntimeError('Output format not recognized: %s' % output_type)

sim = Simulation()
#    @check_class(phd.IntegrateBase)
#    def set_integrator(self, integrator):
#        """
#        Set integrator to evolve the simulation.
#        """
#        self.integrator = integrator
#
#    def solve(self):
#        """
#        Main driver to evolve the equations. Responsible for advancing
#        the simulation while outputting data to disk at appropriate
#        times.
#        """
#        self.start_up_message()
#        self.integrate.initialize()
#
#        # output initial state of simulation
#        self.logs('info', 'Writting initial output')
#        self.output_data()
#
#        # evolve the simulation
#        self.integrate.before_loop(self)
#        self.logs('Beginning integration loop')
#        while not self.integrator.finished():
#
#            self.logs('info', 'Starting iteration: %d time: %f dt: %f' %\
#                    (self.integrator.iteration,
#                     self.integrator.time,
#                     self.integrator.dt))
#
#            # advance one time step
#            self.integrator.evolve_timestep()
#            self.logs('success', 'Finished iteration: %d time: %f dt: %f' %\
#                    self.integrator.iteration)
#
#            # compute new time step
#            self.integrator.compute_timestep()
#            self.modify_timestep() # if needed
#
#            # output if needed
#            if self.check_for_output():
#                self.logs('info',
#                        'Writting output at time %g, iteration %d, dt %g: %s' %\
#                        (self.integrator.iteration,
#                        self.integrator.time,
#                        self.integrator.dt))
#                self.ouput_data()
#
#        self.after_loop(self)
#        self.logs('success', 'Simulation succesfully finished!')
#
#    def start_up_message(self):
#        message = logo_str
#        message += '\nSolving %s problem' % self.simulation_name
#
#        if self.parallel_run:
#            message += '\nRunning in parallel: number of processors %d' %\
#                    self.num_procs
#        else:
#            message += '\nRunning in serial'
#
#        message += '\nOutput data will be saved at: %s' % self.output_dir
#
#        # log message
#        self.log('info', message)
#
#    def modify_timestep(self):
#        '''
#        Compute time step for the next iteration. First the integrator time step
#        is called. Then it is modified by the simulation as to ensure it is
#        constrained.
#        '''
#
#        dt = self.integrator.dt
#        if self.integrator.iteration == 0:
#            # shrink if first iteration
#            dt = self.initial_timestep_factor*dt
#        else:
#            # constrain rate of change
#            dt = min(max_dt_change*self.old_dt, dt)
#        self.old_dt = dt
#
#        # ensure the simulation stops at final time
#        if self.integrator.time + dt > self.integrator.final_time:
#            dt = self.final_time - self.integrator.time
#        self.integrator.set_dt(dt)
#
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
#
#    def barrier(self):
#        if self.parallel_run:
#            self.comm.barrier()
#
#    def logs(log_type, msg):
#        if self.rank == 0:
#            if log_type == 'debug':
#                phdLogger.debug(msg)
#            elif log_type == 'info':
#                phdLogger.info(msg)
#            elif log_type == 'success':
#                phdLogger.success(msg)
#            elif log_type == 'warning':
#                phdLogger.warning(msg)
#            elif log_type == 'error':
#                phdLogger.error(msg)
#            elif log_type == 'critical':
#                phdLogger.critical(msg)
