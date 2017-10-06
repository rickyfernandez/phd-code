import os
import json
import h5py
import numpy as np
from mpi4py import MPI

import phd
from ..utils.particle_tags import ParticleTAGS


class Simulation(object):
    """Marshalls the simulation."""
    def __init__(
        self, tf=1.0, cfl=0.5, pfreq=100, tfreq=100., relax_num_iterations=0, output_relax=False,
        fname='simulation', outdir=None):
        """Constructor

        Parameters:
        -----------

        mesh : phd.mesh.mesh
           tesselation

        tf : double
            Final time

        pfreq : int
            Output printing frequency

        tfreq : double
            Output time frequency

        fname : str
            Output file base name

        iteration_count : int
            Simulation iteration counter. Initialize with non-zero for a restart

        current_time : double
            Simulation time. Initialize with non-zero for a restart
        """
        self.pc = self.mesh = self.domain = self.riemann = None
        self.boundary = self.integrator = self.reconstruction = None

        self.cfl = cfl
        self.pfreq = pfreq
        self.tfreq = tfreq
        self.tf = tf

        self.output = 0
        self.output_relax = output_relax

        # iteration iteration_counter and time
        self.relax_num_iterations = relax_num_iterations
        self.iteration_count = 0
        self.current_time = 0.0

        self.fname = fname

        if not outdir:
            outdir = self.fname + '_output'

        # save the path where we want to dump output
        self.path = os.path.abspath(outdir)
        os.makedirs(self.path)

    def add_particles(self, cl):
        if isinstance(cl, phd.CarrayContainer):
            self.pc = cl
        else:
            raise RuntimeError("%s component not type CarrayContainer" % cl.__class__.__name__)

    def add_domain(self, cl):
        if isinstance(cl, phd.DomainLimits):
            self.domain = cl
        else:
            raise RuntimeError("%s component not type DomainLimits" % cl.__class__.__name__)

    def add_boundary(self, cl):
        if isinstance(cl, phd.Boundary):
            self.boundary = cl
        else:
            raise RuntimeError("%s component not type Boundary" % cl.__class__.__name__)

    def add_mesh(self, cl):
        if isinstance(cl, phd.Mesh):
            self.mesh = cl
        else:
            raise RuntimeError("%s component not type Mesh" % cl.__class__.__name__)

    def add_reconstruction(self, cl):
        if isinstance(cl, phd.ReconstructionBase):
            self.reconstruction = cl
        else:
            raise RuntimeError("%s component not type ReconstructionBase" % cl.__class__.__name__)

    def add_riemann(self, cl):
        if isinstance(cl, phd.RiemannBase):
            self.riemann = cl
        else:
            raise RuntimeError("%s component not type RiemannBase" % cl.__class__.__name__)

    def add_integrator(self, cl):
        if isinstance(cl, phd.IntegrateBase):
            self.integrator = cl
        else:
            raise RuntimeError("%s component not type IntegratorBase" % cl.__class__.__name__)

    def _create_components_from_dict(self, cl_dict):
        for comp_name, (cl_name, cl_param) in cl_dict.iteritems():
            cl = getattr(phd, cl_name)
            x = cl(**cl_param)
            setattr(self, comp_name, x)

    def _create_components_timeshot(self):
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

        for attr_name, cl in self.__dict__.iteritems():
            comp = getattr(self, attr_name)
            if isinstance(comp, (int, float, str)):
                continue
            if comp == None:
                raise RuntimeError("Component: %s not set." % attr_name)

    def _initialize(self):

        self._check_component()

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
        """Main solver"""

        self._initialize()

        tf = self.tf
        mesh = self.mesh; boundary = self.boundary; integrator = self.integrator
        current_time = self.current_time; iteration_count = self.iteration_count
        domain = self.domain; pc = self.pc

        # create initial tessellation - including ghost particles
        mesh.build_geometry(pc)

        if self.relax_num_iterations != 0:
            self._relax_geometry(self.relax_num_iterations)

        # convert primitive values to conserative
        integrator.conserative_from_primitive()

        # main solver iteration
        time_counter = dt = 0.0
        while current_time < tf:

            mesh.build_geometry(pc)
            integrator.primitive_from_conserative()

            # I/O
            if iteration_count % self.pfreq == 0:
                self._save(iteration_count, current_time, dt)

            # calculate the time step and adjust if necessary
            dt = self.cfl*integrator.compute_time_step()

            if (current_time + dt > tf ):
                dt = tf - current_time

            if ( (time_counter + dt) > self.tfreq ):
                dt = self.tfreq - time_counter
                self._save(iteration_count, current_time+dt, dt)
                time_counter -= dt

            # integrate with the corrected time step
            integrator.integrate(dt, current_time, iteration_count)

            iteration_count += 1; current_time += dt
            time_counter += dt

            boundary.migrate_boundary_particles(pc)

        mesh.build_geometry(pc)
        integrator.primitive_from_conserative()

        # final output
        self._save(iteration_count, current_time, dt)
        self.current_time = current_time

    def _save(self, iteration_count, current_time, dt):

        f = h5py.File(self.path + '/' + self.fname + '_' + `self.output`.zfill(4) + '.hdf5', 'w')
        for prop in self.pc.properties.keys():
            f["/" + prop] = self.pc[prop]

        f.attrs['iteration_count'] = iteration_count
        f.attrs['time'] = current_time
        f.attrs['dt'] = dt
        f.close()

        self.output += 1

#    def _compute_primitives(self):
#        pc = self.pc
#
#        vol  = pc['volume']
#        mass = pc['mass']
#        ener = pc['energy']
#
#        # update primitive variables
#        velocity_sq = 0
#        pc['density'][:] = mass/vol
#        for axis in self.dimensions:
#            pc['velocity-' + axis][:] = pc['momentum-' + axis]/mass
#            velocity_sq += pc['velocity-' + axis]**2
#
#        pc['pressure'][:] = (ener/vol - 0.5*pc['density']*velocity_sq)*(self.gamma-1.0)
#
#
#    def _set_initial_state_from_primitive(self):
#        pc = self.pc
#
#        vol  = pc['volume']
#        mass = pc['density']*vol
#
#        velocity_sq = 0
#        pc['mass'][:] = mass
#        for axis in self.dimensions:
#            pc['momentum-' + axis][:] = pc['velocity-' + axis]*mass
#            velocity_sq += pc['velocity-' + axis]**2
#
#        pc['energy'][:] = (0.5*pc['density']*velocity_sq + pc['pressure']/(self.gamma-1.0))*vol


    def _relax_geometry(self, num_iterations):
        for i in range(num_iterations):

            # move real particles towards center of mass
            real = self.pc['tag'] == ParticleTAGS.Real
            for axis in self.dimensions:
                self.pc['position-' + axis][real] += self.pc['dcom-' + axis][real]

            # some real particles may have left domain
            self.boundary.migrate_boundary_particles(self.pc)

            self.mesh.build_geometry(self.pc)
            if self.output_relax:
                self._save(-1, self.current_time, 0)

class SimulationParallel(Simulation):
    """Marshalls simulation in parallel."""
    def __init__(
        self, load_balance_freq=5, tf=1.0, cfl=0.5, pfreq=100, tfreq=100,
        relax_num_iterations=0, output_relax=False, fname='simulation',
        outdir=None, iteration_count=0, current_time=0):
        """Constructor

        Parameters:
        -----------

        mesh : phd.mesh.mesh
           tesselation

        tf, dt : double
            Final time and default time-step

        pfreq : int
            Output printing frequency

        tfreq : int
            Output time frequency

        fname : str
            Output file base name

        iteration_count : int
            Simulation iteration counter. Initialize with non-zero for a restart

        current_time : double
            Simulation time. Initialize with non-zero for a restart

        conservation_check : bool
            Perform total energy check at start and end
        """
        self.pc = self.mesh = self.domain = self.riemann = None
        self.boundary = self.integrator = self.reconstruction = None
        self.comm = self.load_balance = None

        self.cfl = cfl
        self.pfreq = pfreq
        self.tfreq = tfreq
        self.tf = tf

        self.output = 0
        self.output_relax = output_relax

        # iteration iteration_counter and time
        self.relax_num_iterations = relax_num_iterations
        self.iteration_count = iteration_count
        self.current_time = current_time

        self.fname = fname

        if not outdir:
            outdir = self.fname + '_output'

        #save the path where we want to dump output
        self.path = os.path.abspath(outdir)

        #self.load_balance = load_balance
        self.load_balance_freq = load_balance_freq

    def _initialize(self):

        self._check_component()

        # distribute particles among processors
        self.load_balance.comm = self.comm
        self.load_balance.domain = self.domain
        self.load_balance._initialize()

        # boundary conditions - relect, periodic, or mixture
        self.boundary.comm = self.comm
        self.boundary.domain = self.domain
        self.boundary.load_bal = self.load_balance
        self.boundary._initialize()

        # algorithm to construct voronoi mesh
        self.mesh.boundary = self.boundary
        self.mesh._initialize()

        # create states at face for riemann solver
        self.reconstruction.pc = self.pc
        self.reconstruction.mesh = self.mesh
        self.reconstruction._initialize()

        # riemann solver
        self.riemann.reconstruction = self.reconstruction

        # how are the equations advance in time
        self.integrator.pc = self.pc
        self.integrator.mesh = self.mesh
        self.integrator.riemann = self.riemann
        self.integrator._initialize()

        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.gamma = self.integrator.riemann.gamma
        self.dimensions = 'xyz'[:self.mesh.dim]

        if self.rank == 0:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
        self.comm.barrier()

    def add_communicator(self, cl):
        if isinstance(cl, MPI.Intracomm):
            self.comm = cl
        else:
            raise RuntimeError("%s component not type Intracomm" % cl.__class__.__name__)

    def add_loadbalance(self, cl):
        if isinstance(cl, phd.LoadBalance):
            self.load_balance = cl
        else:
            raise RuntimeError("%s component not type LoadBalance" % cl.__class__.__name__)

    def solve(self, **kwargs):
        """Main solver"""

        self._initialize()

        tf = self.tf
        mesh = self.mesh; boundary = self.boundary; integrator = self.integrator
        current_time = self.current_time; iteration_count = self.iteration_count
        domain = self.domain; pc = self.pc; load_balance = self.load_balance
        comm = self.comm

        local_dt  = np.zeros(1)
        global_dt = np.zeros(1)

        # create initial tessellation
        load_balance.decomposition(pc)

        # create initial tessellation - including ghost particles
        mesh.build_geometry(pc)

        if self.relax_num_iterations != 0:
            self._relax_geometry(self.relax_num_iterations)

        # convert primitive values to conserative
        #self._set_initial_state_from_primitive()
        integrator.conserative_from_primitive()

        # main solver iteration
        time_counter = dt = 0.0
        while current_time < tf:

            # check if load balance is needed
            if iteration_count % self.load_balance_freq == 0:
                load_balance.decomposition(pc)

            mesh.build_geometry(pc)
            #self._compute_primitives()
            integrator.primitive_from_conserative()

            # I/O
            if iteration_count % self.pfreq == 0:
                self._save(iteration_count, current_time, dt)

            # calculate the time step and adjust if necessary this has to be a mpi call
            local_dt[0] = self.cfl*integrator.compute_time_step()
            comm.Allreduce(sendbuf=local_dt, recvbuf=global_dt, op=MPI.MIN)
            dt = global_dt[0]

            if (current_time + dt > tf):
                dt =  tf - current_time

            if self.rank == 0:
                print 'iteration:', iteration_count, 'time:', current_time, 'dt:', dt

            if ( (time_counter + dt) > self.tfreq ):
                dt = self.tfreq - time_counter
                self._save(iteration_count, current_time+dt, dt)
                time_counter = -dt

            # integrate with the corrected time step
            integrator.integrate(dt, current_time, iteration_count)
            iteration_count += 1; current_time += dt
            time_counter += dt

            boundary.migrate_boundary_particles(pc)

        mesh.build_geometry(pc)
        #self._compute_primitives()
        integrator.primitive_from_conserative()

        # final output
        self._save(iteration_count, current_time, dt)
        self.current_time = current_time

    def _save(self, iteration_count, current_time, dt):

        output_dir = self.path + "/" + self.fname + "_" + `self.output`.zfill(4)

        if self.rank == 0:
            #if not os.path.isdir(self.path):
            os.mkdir(output_dir)
        self.comm.barrier()

        f = h5py.File(output_dir + "/" + "data" + `self.output`.zfill(4)
                + '_cpu' + `self.rank`.zfill(4) + ".hdf5", "w")

        for prop in self.pc.properties.keys():
            f["/" + prop] = self.pc[prop]

        f.attrs["iteration_count"] = iteration_count
        f.attrs["time"] = current_time
        f.attrs["dt"] = dt
        f.close()

        self.output += 1

    def _relax_geometry(self, num_iterations):
        for i in range(num_iterations):

            # move real particles towards center of mass
            real = self.pc['tag'] == ParticleTAGS.Real
            for axis in self.dimensions:
                self.pc['position-' + axis][real] += self.pc['dcom-' + axis][real]

            # some real particles may have left domain
            self.boundary.migrate_boundary_particles(self.pc)
            self.load_balance.decomposition(self.pc)

            # generate new ghost particles
            self.mesh.build_geometry(self.pc)
            if self.output_relax:
                self._save(-1, self.current_time, 0)
