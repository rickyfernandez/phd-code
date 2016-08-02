import h5py
import numpy as np
from mpi4py import MPI
import os

from ..utils.particle_tags import ParticleTAGS

# for debug plotting 
import matplotlib.pyplot as plt


class Solver(object):
    """Solver object that marshalls the simulation."""
    def __init__(
        self, integrator, tf=1.0, dt=1e-3, cfl=0.5, pfreq=100, tfreq=100., relax_num_iterations=0, output_relax=False,
        fname='simulation', outdir=None, iteration_count=0, current_time=0.):
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
            Solver iteration counter. Initialize with non-zero for a restart

        current_time : double
            Solver time. Initialize with non-zero for a restart
        """
        self.mesh = integrator.mesh
        self.pc = integrator.particles
        self.boundary = integrator.mesh.boundary
        self.domain = integrator.mesh.boundary.domain

        self.dimensions = 'xyz'[:self.mesh.dim]

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

        # default integrator
        self.integrator = integrator
        self.gamma = integrator.riemann.gamma

        if not outdir:
            outdir = self.fname + '_output'

        #save the path where we want to dump output
        self.path = os.path.abspath(outdir)
        os.makedirs(self.path)

    def solve(self):
        """Main solver"""

        tf = self.tf
        mesh = self.mesh; boundary = self.boundary; integrator = self.integrator
        current_time = self.current_time; iteration_count = self.iteration_count
        domain = self.domain; pc = self.pc

        if self.relax_num_iterations != 0:
            self._relax_geometry(self.relax_num_iterations)

        # create initial tessellation - including ghost particles
        mesh.build_geometry(pc)

        # convert primitive values to conserative
        self._set_initial_state_from_primitive()

        # main solver iteration
        time_counter = dt = 0.0
        while current_time < tf:

            mesh.build_geometry(pc)
            self.compute_primitives()

            # I/O
            if iteration_count % self.pfreq == 0:
                self.save(iteration_count, current_time, dt)

            # calculate the time step and adjust if necessary
            dt = self.cfl*integrator.compute_time_step()

            if (current_time + dt > tf ):
                dt = tf - current_time

            print 'iteration:', iteration_count, 'time:', current_time, 'dt:', dt

            if ( (time_counter + dt) > self.tfreq ):
                dt = self.tfreq - time_counter
                self.save(iteration_count, current_time+dt, dt)
                time_counter -= dt

            # integrate with the corrected time step
            integrator.integrate(dt, current_time, iteration_count)

            iteration_count += 1; current_time += dt
            time_counter += dt

        mesh.build_geometry(pc)
        self.compute_primitives()

        # final output
        self.save(iteration_count, current_time, dt)
        self.current_time = current_time

    def save(self, iteration_count, current_time, dt):

        f = h5py.File(self.path + "/" + self.fname + `self.output`.zfill(4) + ".hdf5", "w")
        for prop in self.pc.properties.keys():
            f["/" + prop] = self.pc[prop]

        f.attrs["iteration_count"] = iteration_count
        f.attrs["time"] = current_time
        f.attrs["dt"] = dt
        f.close()

        self.output += 1

    def compute_primitives(self):
        pc = self.pc

        vol  = pc['volume']
        mass = pc['mass']
        ener = pc['energy']

        # update primitive variables
        velocity_sq = 0
        pc['density'][:] = mass/vol
        for axis in self.dimensions:
            pc['velocity-' + axis][:] = pc['momentum-' + axis]/mass
            velocity_sq += pc['velocity-' + axis]**2

        pc['pressure'][:] = (ener/vol - 0.5*pc['density']*velocity_sq)*(self.gamma-1.0)


    def _set_initial_state_from_primitive(self):
        pc = self.pc

        vol  = pc['volume']
        mass = pc['density']*vol

        velocity_sq = 0
        pc['mass'][:] = mass
        for axis in self.dimensions:
            pc['momentum-' + axis][:] = pc['velocity-' + axis]*mass
            velocity_sq += pc['velocity-' + axis]**2

        pc['energy'][:] = (0.5*pc['density']*velocity_sq + pc['pressure']/(self.gamma-1.0))*vol


    def _relax_geometry(self, num_iterations):

        lo = self.domain[0,0]
        hi = self.domain[1,0]

        for i in range(num_iterations):

            self.mesh.build_geometry(self.pc)

            # move real particles towards center of mass
            real = self.pc['tag'] == ParticleTAGS.Real
            for axis in self.dimensions:
                self.pc['position-' + axis][real] += self.pc['dcom-' + axis][real]

            # some real particles may have left the domain, need to relabel
            indices = (((lo <= self.pc['position-x']) & (self.pc['position-x'] <= hi)) \
                     & ((lo <= self.pc['position-y']) & (self.pc['position-y'] <= hi)))
                     #& ((lo <= self.pc['position-z']) & (self.pc['position-z'] <= hi)))
            self.pc['tag'][indices]  = ParticleTAGS.Real
            self.pc['tag'][~indices] = ParticleTAGS.Ghost

            # generate new ghost particles
            self.mesh.build_geometry(self.pc)
            if self.output_relax:
                self.save(-1, self.current_time, 0)

class SolverParallel(object):
    """Solver object that marshalls the simulation in parallel."""
    def __init__(
        self, integrator, load_balance, comm=None, tf=1.0, dt=1e-3, cfl=0.5, pfreq=100, tfreq=100,
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
            Solver iteration counter. Initialize with non-zero for a restart

        current_time : double
            Solver time. Initialize with non-zero for a restart

        conservation_check : bool
            Perform total energy check at start and end
        """
        self.mesh = integrator.mesh
        self.pc = integrator.particles
        self.boundary = integrator.mesh.boundary
        self.domain = integrator.mesh.boundary.domain

        self.dimensions = 'xyz'[:self.mesh.dim]

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

        # default integrator
        self.integrator = integrator
        self.gamma = integrator.riemann.gamma

        if not outdir:
            outdir = self.fname + '_output'

        #save the path where we want to dump output
        self.path = os.path.abspath(outdir)

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.load_balance = load_balance

        if self.rank == 0:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
        self.comm.barrier()

    def solve(self, **kwargs):
        """Main solver"""

        tf = self.tf
        mesh = self.mesh; boundary = self.boundary; integrator = self.integrator
        current_time = self.current_time; iteration_count = self.iteration_count
        domain = self.domain; pc = self.pc; load_balance = self.load_balance
        comm = self.comm

        if self.relax_num_iterations != 0:
            self._relax_geometry(self.relax_num_iterations)

        local_dt  = np.zeros(1)
        global_dt = np.zeros(1)

        # create initial tessellation
        load_balance.decomposition(pc)

        # create initial tessellation - including ghost particles
        mesh.build_geometry(pc)

        # convert primitive values to conserative
        self._set_initial_state_from_primitive()

        # main solver iteration
        time_counter = dt = 0.0
        while current_time < tf:
        #for i in range(13):

            # check if load balance is needed
            #if self.load_balance.check():
            #    load_balance.decomposition()

            mesh.build_geometry(pc)
            self.compute_primitives()

            # I/O
            if iteration_count % self.pfreq == 0:
                self.save(iteration_count, current_time, dt)

            # calculate the time step and adjust if necessary this has to be a mpi call
            local_dt[0] = self.cfl*integrator.compute_time_step()
            comm.Allreduce(sendbuf=local_dt, recvbuf=global_dt, op=MPI.MIN)
            dt = global_dt[0]

            if (current_time + dt > tf ):
                dt =  tf - current_time

            if self.rank == 0:
                print 'iteration:', iteration_count, 'time:', current_time, 'dt:', dt

            if ( (time_counter + dt) > self.tfreq ):
                dt = self.tfreq - time_counter
                self.save(iteration_count, current_time+dt, dt)
                time_counter = -dt

            # integrate with the corrected time step
            integrator.integrate(dt, current_time, iteration_count)
            iteration_count += 1; current_time += dt
            time_counter += dt

            boundary.migrate_boundary_particles(pc)

            if self.rank == 3:
                plt.clf()
                #flag = pc['tag'] == 25
                real = pc['tag'] == 0
                #plt.scatter(pc['position-x'][flag], pc['position-y'][flag], color='red')
                plt.scatter(pc['position-x'][real], pc['position-y'][real], color='blue')
                plt.axvline(x=0.5)
                plt.xlim(-0.1,1.1)
                plt.ylim(-0.1,1.1)
                plt.savefig('test_scatter_' + `iteration_count`.zfill(2) + '.pdf')


        mesh.build_geometry(pc)
        self.compute_primitives()

        # final output
        self.save(iteration_count, current_time, dt)
        self.current_time = current_time

    def save(self, iteration_count, current_time, dt):

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

    def compute_primitives(self):
        pc = self.pc

        vol  = pc['volume']
        mass = pc['mass']
        ener = pc['energy']

        # update primitive variables
        velocity_sq = 0
        pc['density'][:] = mass/vol
        for axis in self.dimensions:
            pc['velocity-' + axis][:] = pc['momentum-' + axis]/mass
            velocity_sq += pc['velocity-' + axis]**2

        pc['pressure'][:] = (ener/vol - 0.5*pc['density']*velocity_sq)*(self.gamma-1.0)


    def _set_initial_state_from_primitive(self):
        pc = self.pc

        vol  = self.pc['volume']
        mass = self.pc['density']*vol

        velocity_sq = 0
        pc['mass'][:] = mass
        for axis in self.dimensions:
            pc['momentum-' + axis][:] = pc['velocity-' + axis]*mass
            velocity_sq += pc['velocity-' + axis]**2

        pc['energy'][:] = (0.5*pc['density']*velocity_sq + pc['pressure']/(self.gamma-1.0))*vol


    def _relax_geometry(self, num_iterations):

        lo = self.domain[0,0]
        hi = self.domain[1,0]

        for i in range(num_iterations):

            self.mesh.build_geometry(self.pc)

            # move real particles towards center of mass
            real = self.pc['tag'] == ParticleTAGS.Real
            for axis in self.dimensions:
                self.pc['position-' + axis][real] += self.pc['dcom-' + axis][real]

            # some real particles may have left the domain, need to relabel
            indices = (((lo <= self.pc['position-x']) & (self.pc['position-x'] <= hi)) \
                     & ((lo <= self.pc['position-y']) & (self.pc['position-y'] <= hi)))
                     #& ((lo <= self.pc['position-z']) & (self.pc['position-z'] <= hi)))
            self.pc['tag'][indices]  = ParticleTAGS.Real
            self.pc['tag'][~indices] = ParticleTAGS.Ghost

            self.load_balance.migrate_boundary_particles(pc)

            # generate new ghost particles
            self.mesh.build_geometry(self.pc)
            if self.output_relax:
                self.save(-1, self.current_time, 0)
