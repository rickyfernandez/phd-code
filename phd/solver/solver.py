import h5py
import numpy as np
from mpi4py import MPI
import os

from utils.particle_tags import ParticleTAGS

# for debug plotting 
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib

def vor_plot2(pc, mesh, iteration_count):

    # debugging plot --- turn to a routine later ---
    l = []
    ii = 0; jj = 0
    for i in range(pc.get_number_of_particles()):

        jj += mesh['number of neighbors'][i]*2

        if pc['tag'][i] == ParticleTAGS.Real or pc['type'][i] == ParticleTAGS.Boundary:

            verts_indices = np.unique(mesh['faces'][ii:jj])
            verts = mesh['voronoi vertices'][verts_indices]

            # coordinates of neighbors relative to particle p
            xc = verts[:,0] - pc['position-x'][i]
            yc = verts[:,1] - pc['position-y'][i]

            # sort in counter clock wise order
            sorted_vertices = np.argsort(np.angle(xc+1j*yc))
            verts = verts[sorted_vertices]

            l.append(Polygon(verts, True))

        ii = jj

    dens = pc['density']

    # add colormap
    colors = []
    for i in range(pc.get_number_of_particles()):
        if pc['tag'][i] == ParticleTAGS.Real or pc['type'][i] == ParticleTAGS.Boundary:
            colors.append(dens[i])

    fig, ax = plt.subplots()
    p = PatchCollection(l, alpha=0.4)
    p.set_array(np.array(colors))
    p.set_clim([0, 4])

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.add_collection(p)
    ax.set_aspect('equal')

    plt.colorbar(p)
    plt.savefig('test_'+ `iteration_count`.zfill(4) + '.pdf', format='pdf')

    plt.cla()
    plt.clf()

##def vor_plot(pc, mesh, rank, load_balance):
def vor_plot(pc, mesh, rank):

    # debugging plot --- turn to a routine later ---
    l = []
    ii = 0; jj = 0
    for i in range(pc.get_number_of_particles()):

        jj += mesh['number of neighbors'][i]*2

        if pc['tag'][i] == ParticleTAGS.Real or pc['type'][i] == ParticleTAGS.Boundary or pc['type'][i] == ParticleTAGS.BoundarySecond:

            verts_indices = np.unique(mesh['faces'][ii:jj])
            verts = mesh['voronoi vertices'][verts_indices]

            # coordinates of neighbors relative to particle p
            xc = verts[:,0] - pc['position-x'][i]
            yc = verts[:,1] - pc['position-y'][i]

            # sort in counter clock wise order
            sorted_vertices = np.argsort(np.angle(xc+1j*yc))
            verts = verts[sorted_vertices]

            l.append(Polygon(verts, True))

        ii = jj

    colors = []
    for i in range(pc.get_number_of_particles()):
        if pc['tag'][i] == ParticleTAGS.Real or pc['type'][i] == ParticleTAGS.Boundary or pc['type'][i] == ParticleTAGS.BoundarySecond:
            colors.append(pc["density"][i])

    fig, ax = plt.subplots()
    p = PatchCollection(l, alpha=0.4)
    p.set_array(np.array(colors))
    p.set_clim([0, 4])

    ax.set_xlim(-.1,1.1)
    ax.set_ylim(-.1,1.1)
    ax.add_collection(p)
    ax.set_aspect('equal')

#    current_axis = plt.gca()
#    for node in load_balance.global_tree.dump_data():
#        x = node[0]/2.0**load_balance.order
#        y = node[1]/2.0**load_balance.order
#        w = node[2]/2.0**load_balance.order
#        current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, fill=None))

    tag = pc['type']
#    ghost = tag == 0
#    plt.scatter(pc['position-x'][ghost], pc['position-y'][ghost], marker=".", color='lightsteelblue')
    exterior = tag == 2
    plt.scatter(pc['position-x'][exterior], pc['position-y'][exterior], marker=".", color='red')
    exterior = tag == 8
    plt.scatter(pc['position-x'][exterior], pc['position-y'][exterior], marker=".", color='cyan')


    plt.savefig("plot_init_proc_%d.pdf" % rank, format='pdf')
    plt.clf()

class Solver(object):
    """Solver object that marshalls the simulation."""
    def __init__(
        self, mesh, integrator, boundary, domain, tf=1.0, dt=1e-3, cfl=0.5, pfreq=100, tfreq=100., fname='simulation',
        outdir=None, iteration_count=0, current_time=0.):
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
        self.mesh = mesh
        self.pc = mesh.particles
        self.boundary = boundary
        self.domain = domain

        self.cfl = cfl

        self.pfreq = pfreq
        self.tfreq = tfreq
        self.tf = tf

        self.output = 0

        # iteration iteration_counter and time
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

        # create initial tessellation
        mesh.tessellate()
        mesh.update_boundary_particles()
        mesh.update_second_boundary_particles()
        mesh.compute_cell_info()

        # convert primitive values to conserative
        self._set_initial_state_from_primitive()

        # main solver iteration
        time_counter = dt = 0.0
        while current_time < tf:

            boundary.update_ghost_particles(pc, mesh, domain)
            mesh.build_geometry(self.gamma)

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

        boundary.update_ghost_particles(pc, mesh, domain)
        mesh.build_geometry(self.gamma)
        vor_plot2(pc, mesh, iteration_count)

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

    def _set_initial_state_from_primitive(self):

        vol  = self.pc['volume']
        mass = self.pc['density']*vol

        self.pc['mass'][:] = mass
        self.pc['momentum-x'][:] = self.pc['velocity-x']*mass
        self.pc['momentum-y'][:] = self.pc['velocity-y']*mass

        self.pc['energy'][:] = (0.5*self.pc['density']*(self.pc['velocity-x']**2 + self.pc['velocity-y']**2) +\
                self.pc['pressure']/(self.gamma-1.0))*vol


class SolverParallel(object):
    """Solver object that marshalls the simulation in parallel."""
    def __init__(
        self, mesh, integrator, boundary, domain, load_balance, comm=None, tf=1.0, dt=1e-3, cfl=0.5, pfreq=100, tfreq=100, fname='simulation',
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
        self.mesh = mesh
        self.pc = mesh.particles
        self.boundary = boundary
        self.domain = domain

        self.cfl = cfl

        self.pfreq = pfreq
        self.tfreq = tfreq
        self.tf = tf

        self.output = 0

        # iteration iteration_counter and time
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

        local_dt  = np.zeros(1)
        global_dt = np.zeros(1)

        # create initial tessellation
        load_balance.decomposition()
        boundary.create_ghost_particles(pc, mesh, domain, load_balance, comm)

        mesh.tessellate()
        mesh.update_boundary_particles()
        mesh.update_second_boundary_particles()
        mesh.compute_cell_info()

        # convert primitive values to conserative
        self._set_initial_state_from_primitive()

        # main solver iteration
        time_counter = dt = 0.0
        while current_time < tf:

            if self.rank == 0:
                print 'iteration:', iteration_count

            boundary.update_ghost_particles(pc, mesh, domain, load_balance, comm)
            mesh.build_geometry(self.gamma)
            #vor_plot(pc, mesh, self.rank)

#k            # check if load balance is needed
#k            if self.load_balance.check():
#k                # create new load balance with ghost and mesh
#k                self.load_balance.load()
#k                boundary.create_ghost_particles(pc, mesh, domain, load_balance, comm)
#k                mesh.build_geometry(self.gamma)
#k            else:
#k                # set new ghost particles and create mesh
#k                boundary.update_ghost_particles(pc, mesh, domain, load_balance, comm)
#k                mesh.build_geometry(self.gamma)

#            # for debug - delete
#            vor_plot(pc, mesh, iteration_count)
#
#            # I/O
            #if iteration_count % self.pfreq == 0:
            #    self.save(iteration_count, current_time, dt)
#
            # calculate the time step and adjust if necessary this has to be a mpi call
            local_dt[0] = integrator.compute_time_step()*self.cfl
            comm.Allreduce(sendbuf=local_dt, recvbuf=global_dt, op=MPI.MIN)
            dt = global_dt[0]

            if (current_time + dt > tf ):
                dt =  tf - current_time

        #    print 'rank:', self.rank, 'local:', local_dt, 'global_dt:', global_dt, 'dt:', dt, current_time

            if ( (time_counter + dt) > self.tfreq ):
                dt = self.tfreq - time_counter
                self.save(iteration_count, current_time+dt, dt)
                time_counter = -dt

            # integrate with the corrected time step
            integrator.integrate(dt, current_time, iteration_count)

            iteration_count += 1; current_time += dt
            time_counter += dt

        # final output
        boundary.update_ghost_particles(pc, mesh, domain, load_balance, comm)
        mesh.build_geometry(self.gamma)
        vor_plot(pc, mesh, self.rank)
#        vor_plot(pc, mesh, self.rank, load_balance)

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

    def _set_initial_state_from_primitive(self):

        vol  = self.pc['volume']
        mass = self.pc['density']*vol

        self.pc['mass'][:] = mass
        self.pc['momentum-x'][:] = self.pc['velocity-x']*mass
        self.pc['momentum-y'][:] = self.pc['velocity-y']*mass

        self.pc['energy'][:] = (0.5*self.pc['density']*(self.pc['velocity-x']**2 + self.pc['velocity-y']**2) +\
                self.pc['pressure']/(self.gamma-1.0))*vol

