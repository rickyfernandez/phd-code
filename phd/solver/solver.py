import h5py
import numpy as np
from os.path import abspath

class Solver(object):
    """Solver object that marshalls the simulation."""
    def __init__(
        self, mesh, integrator, tf=1.0, dt=1e-3, cfl=0.5, pfreq=100, tfreq=1, fname='simulation',
        iteration_count=0, current_time=0, conservation_check=True):
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
        self.mesh = mesh
        self.pa = mesh.pa

        self.cfl = cfl

        self.pfreq = pfreq
        self.tfreq = tfreq
        self.tf = tf
        self.dt = dt

        # iteration iteration_counter and time
        self.iteration_count = iteration_count
        self.current_time = current_time

        self.fname = fname

        # default integrator
        self.integrator = integrator
        self.integrator.func.set_cfl(cfl)
        self.integrator.func.set_final_time(tf)

        if not outdir:
            outdir = self.fname + '_output'

        #save the path where we want to dump output
        self.path = abspath(outdir)

        if rank == 0:
            import os
            os.makedirs(self.path)


    def solve(self, **kwargs):
        """Main solver"""

        dt = self.dt
        pa = self.pa; nnps = self.nnps; integrator = self.integrator
        current_time = self.current_time; iteration_count = self.iteration_count

        # main solver iteration
        time_counter = 0.0
        tf = self.tf
        while current_time < tf:

            # check if load balance is needed
            if self.load_balance.check():
                # create new load balance with ghost and mesh
                self.load_balance.load()
                self.boundary.create_ghost_particles()
                self.mesh.tessellate()
            else:
                # set new ghost particles and create mesh
                self.boundary.create_ghost_particles()
                self.mesh.tessellate()

            # I/O
            if iteration_count % self.pfreq == 0:
                self.save(iteration_count, current_time, dt)

            # calculate the time step and adjust if necessary
            dt = integrator.compute_time_step(dt)

            if (current_time + dt > tf ):
                dt =  tf - current_time

            if ( (time_counter + dt) > self.tfreq ):
                dt = self.tfreq - time_counter
                self.save(iteration_count, current_time+dt, dt)
                time_counter = -dt

            # integrate with the corrected time step
            integrator.integrate(dt, current_time, iteration_count)

            iteration_count += 1; current_time += dt
            time_counter += dt

        # final output
        self.save(iteration_count, current_time, dt)
        self.current_time = current_time

    def save(self, iteration_count, t, dt):

        f = h5py.File(self.path + self.fname + "_" +
                `iteration_count`.zfill(4) + '_cpu' + ``.zfill(4) + ".hdf5", "w")

        for prop in self.pa.properties.keys():
            f["/" + prop] = self.pa[prop]

        f.attrs["iteration_count"] = iteration_count
        f.attrs["time"] = self.time
        f.attrs["dt"] = self.time

        f.close()
