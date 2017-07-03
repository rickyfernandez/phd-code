import os
import json
import h5py
import logging
import numpy as np

import phd

try:
    import mpi4py.MPI as mpi
    _found_mpi = True
except ImportError:
    _found_mpi = False

logger = logging.getLogger(__name__)


class Simulation(object):
    """Marshalls the simulation."""
    def __init__(
        self, final_time=1.0, max_dt_change=1.e33, initial_timestep_factor=1.0, cfl=0.5,
        output_time_interval=0, relax_num_iterations=0, output_relax=False,
        simulation_name='simulation', output_type='hdf5', log_level='debug'):
        """Constructor

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
           Simulation name data writes will be prefiex with the string

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

        self.pc = self.mesh = self.domain = self.riemann = None
        self.boundary = self.integrator = self.reconstruction = None
        self.load_balance = None

        self.final_time = final_time
        self.max_dt_change = max_dt_change
        self.initial_timestep_factor = initial_timestep_factor

        self.cfl = cfl
        self.output_time_interval = output_time_interval

        # mesh relaxation
        self.output_relax = output_relax
        self.relax_num_iterations = relax_num_iterations

        # parallel parameters
        self.comm = None
        self.rank = 0
        self.num_procs = 1
        self.parallel_run = True if _found_mpi else: False

        # if mpi4py is available
        if self.parallel_run:
            self.comm = mpi.COMM_WORLD
            self.num_procs = comm.Get_size()
            self.rank = comm.Get_rank()

        # logger parameters
        self.log_level = log_level
        self.log_filename = self.simulation_name + '.log'
        self.log_format = '%(levelname)s|%(asctime)s|%(name)s|%(message)s'

        # setup logger
        logger.setLevel(self.log_level)
        formatter = logging.Formatter(self.log_format)
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setFormatter(formatter)

        logger.addHanderl(file_handler)
        logger.addHandler(logging.StreamHandler())

        # create direcotry to store outputs
        self.simulation_name = simulation_name
        self.output_directory = self.simulation_name + '_output'

        if output_type == 'hdf5':
            self.output = Hdf5()
        elif output_type = 'npz':
            self.output = Npz():
        else:
            RuntimeError('Ouput format not recognized: %s' % output_type)

    def _initialize_and_link(self):
        """
        """
        self._check_component()

        # load balance for parallel runs
        if self.parallel_run:
            self.load_balance.comm = self.comm
            self.load_balance.domain = self.domain
            self.load_balance._initialize()

        # spatial boundary conditions
        if self.parallel_run:
            self.boundary.comm = self.comm
            self.boundary.load_bal = self.load_balance
        self.boundary.domain = self.domain
        self.boundary._initialize()

        self.mesh.boundary = self.boundary
        self.mesh._initialize()

        self.reconstruction.pc = self.pc
        self.reconstruction.mesh = self.mesh
        self.reconstruction._initialize()

        self.riemann.reconstruction = self.reconstruction

        self.integrator.pc = self.pc
        self.integrator.mesh = self.mesh
        self.integrator.riemann = self.riemann
        self.integrator._initialize()

        self.gamma = self.integrator.riemann.gamma
        self.dimensions = 'xyz'[:self.mesh.dim]

    def solve(self):
        """
        Main driver to evolve the equations. Responsible for advancing
        the simulation while outputting data to disk at appropriate
        times.
        """
        self._initialize()
        self.integrate.before_loop()
        self._output_data()

        # evolve the simulation
        while not self.integrator.finished():

            if self.rank == 0:
                logger.info('iteration: %d time: %f dt: %f' %\
                        (self.integrator.iteration,
                         self.integrator.time,
                         self.integrator.dt))

            # advance one time step
            self.integrator.evolve_timestep()
            self.compute_timestep()

            # output if needed
            if self._check_for_output():
                self._output_data()

        self.after_loop()
        self._output_data()

    def compute_timestep(self):
        '''
        Compute time step for the next iteration. First the integrator time step
        is called. Then it is modified by the simulation as to ensure it is
        constrained.
        '''

        # time step from integrator
        dt = self.integrator.compute_timestep()

        if self.integrator.iteration == 0:
            # shrink if first iteration
            dt = self.initial_timestep_factor*dt
        else:
            # constrain rate of change
            dt = min(max_dt_change*self.old_dt, dt)
        self.old_dt = dt

        # ensure the simulation stops at final time
        if self.integrator.time + dt > self.integrator.final_time:
            dt = self.final_time - self.integrator.time

        self.integrator.set_dt(dt)

    def _output_data(self):
        """
        """
        if self.rank == 0:
            logger.info('Writting output at time %g,iteration %d, dt %g to disk' %\
                    self.integrator.iteration,
                    self.integrator.time
                    self.integrator.dt)

        output_dir = self.path + "/" + self.fname + "_" + `self.output`.zfill(4)

        if self.parallel_run:
            output_dir = output_dir + "/" + "data" + `self.output`.zfill(4)
                + '_cpu' + `self.rank`.zfill(4)

            if self.rank == 0:
                os.mkdir(output_dir)
            self._barrier()

        f = h5py.File(ouput_name + '.hdf5', 'w')
        for prop in self.pc.properties.keys():
            f["/" + prop] = self.pc[prop]

        f.attrs["iteration_count"] = iteration_count
        f.attrs["time"] = current_time
        f.attrs["dt"] = dt
        f.close()

        self.output += 1
        self._barrier()

    def _barrier(self):
        if self.parallel_run:
            self.comm.barrier()

    def _log_message(msg):
        if self.rank == 0:
            logger.info(msg)

    def _create_components_timeshot(self):
        '''
        Cycle through all classes and record all attributes and store
        it in a dictionary, key = class name val = attributes.
        '''
        dict_output = {}
        for attr_name, cl in self.__dict__.iteritems():

            comp = getattr(self, attr_name)

            # ignore parameters
            if isinstance(comp, (int, float, str)):
                continue

            # store components
            d = {}
            for i in dir(cl):
                x = getattr(cl, i)
                if isinstance(x, (int, float, str)):
                    d[i] = x

            dict_output[attr_name] = (cl.__class__.__name__, d)

        return dict_output

    def _check_component(self):
        '''
        Cycle through all classes and make sure all attributes are
        set. If any attributes are not set then raise and exception.
        '''

        for attr_name, cl in self.__dict__.iteritems():
            comp = getattr(self, attr_name)
            if isinstance(comp, (int, float, str)):
                continue
            if comp == None:
                raise RuntimeError("Component: %s not set." % attr_name)

    def _create_components_from_dict(self, cl_dict):
        '''
        Create a simulation from dictionary cl_dict created by _create_components_from_dict
        '''
        for comp_name, (cl_name, cl_param) in cl_dict.iteritems():
            cl = getattr(phd, cl_name)
            x = cl(**cl_param)
            setattr(self, comp_name, x)

class Output(object):
    def __init__(self):
        pass

    def write(file_name, particles, integrator, meta_data):
        pass

    def load(self):
        pass

class HDF5Output(Output):
    def write(self, file_name, integrator):
        with h5py.File(file_name, 'w') as f:

            f.attrs["problem"] = problem_name

            # store current time
            f.attrs["dt"] = integrator.dt
            f.attrs["time"] = integrator.current_time
            f.attrs["iteration_count"] = integrator.iteration_count

            # store particle data
            particle_grp = f.create_group('particles')

            # common information 
            particle_grp.attrs['Real'] = ParticleTAGS.Real
            particle_grp.attrs['Ghost'] = ParticleTAGS.Real
            particle_grp.attrs['number_particles'] = integrator.pc.get_number_particles()

            # store particle data for each field
            for prop_name in pc.properties.keys():
                data_grp = particle_grp.create_group(prop_name)
                data_grp.attrs['dtype'] = integrator.pc.carray_info[prop_name]
                data_grp.create_dataset('data', data=integrator.pc[prop_name])

            # store named groups
            named_grp = f.create_group('named_group')
            for grp, grp_list in pc.named_groups.iteritems():
                named_grp.attrs[grp] = ','.join(grp_list)

    def load(self, file_name):
        with h5py.File(file_name, 'r') as f:

            particle_grp = f['particles']
            num_particles = particles_grp.attrs['number_particles']
            pc = CarrayContainer(num_particles)

            # populate arrays with data
            for field in particle_grp.keys():
                field_grp = particle_grp[field]
                pc.register(num_particles, field, field_grp['dtype'])
                pc[field][:] = field_grp['data'][:]

            particle_grp = f['named_group']
            for grp in named_grp.keys():
                pc.named_gropus[grp] = named_grp[grp].split(',')

        return pc
