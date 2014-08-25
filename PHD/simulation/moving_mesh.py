import h5py
import numpy as np
from PHD.fields import Fields
from PHD.mesh import VoronoiMesh
from PHD.riemann.riemann_base import RiemannBase
from PHD.boundary.boundary_base import BoundaryBase
from PHD.reconstruction.reconstruct_base import ReconstructBase

# for debug plotting 
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib


class MovingMesh(object):

    def __init__(self, gamma = 1.4, CFL = 0.5, max_steps=100, max_time=None, output_cycle = 100000,
            output_name="simulation_", regularization=True):

        # simulation parameters
        self.CFL = CFL
        self.gamma = gamma
        self.max_steps = max_steps
        self.max_time = max_time
        self.output_cycle = output_cycle
        self.output_name = output_name
        self.regularization = regularization

        # particle information
        self.particles = None
        self.fields = None
        self.cell_info = None

        self.particles_index = None
        self.voronoi_vertices = None
        self.neighbor_graph = None
        self.neighbor_graph_sizes = None
        self.face_graph = None
        self.face_graph_sizes = None

        self.time = 0.

        # simulation classes
        self.mesh = VoronoiMesh()
        self.boundary = None
        self.reconstruction = None
        self.riemann_solver = None


    #def get_dt(self, time, prim, vol):
    def get_dt(self):
        """
        Calculate the time step using the CFL condition.
        """

        cells = self.particles_index["real"]
        vol = self.cell_info["volume"]

        dens = self.fields.get_field("density")[cells]
        velx = self.fields.get_field("velocity-x")[cells]
        vely = self.fields.get_field("velocity-x")[cells]
        pres = self.fields.get_field("pressure")[cells]

        # sound speed
        #c = np.sqrt(self.gamma*prim[3,:]/prim[0,:])
        c = np.sqrt(self.gamma*pres/dens)

        # calculate approx radius of each voronoi cell
        R = np.sqrt(vol/np.pi)

        # grab the velocity
        #u = np.sqrt(prim[1,:]**2 + prim[2,:]**2)
        u = np.sqrt(velx**2 + vely**2)
        lam = np.maximum.reduce([np.absolute(u-c), np.absolute(u), np.absolute(u+c)])

        #dt = self.CFL*np.min(R/c)
        dt = self.CFL*np.min(R/lam)

        if self.time + dt > self.max_time:
            dt = self.max_time - self.time

        return dt


#    def _cons_to_prim(self, volume):
#        """
#        Convert volume integrated variables (density, densiy*velocity, Energy) to
#        primitive variables (mass, momentum, pressure).
#        """
#        # conserative vector is mass, momentum, total energy in cell volume
#        mass = self.data[0,:]
#
#        primitive = np.empty(self.data.shape, dtype=np.float64)
#        primitive[0,:] = self.data[0,:]/volume      # density
#        primitive[1:3,:] = self.data[1:3,:]/mass    # velocity
#
#        # pressure
#        primitive[3,:] = (self.data[3,:]/volume-0.5*primitive[0,:]*\
#                (primitive[1,:]**2 + primitive[2,:]**2))*(self.gamma-1.0)
#
#        i = primitive[0,:] < 0
#        if i.any():
#            print "found particles with negative densities"
#            print self.particles[0,i]
#
#        i = primitive[3,:] < 0
#        if i.any():
#            print "found particles with negative pressure"
#            print self.particles[0,i]
#
#        return primitive

    def data_dump(self, num):

        f = h5py.File(self.output_name + "_" + `num`.zfill(4) + ".hdf5", "w")

#        vol  = self.cell_info["volume"]
#        mass = self.data[0,:]
#        vx   = self.data[1,:]/mass
#        vy   = self.data[2,:]/mass
#
#        f["/particles"] = self.particles
#        f["/density"]  = mass/vol
#        f["/velocity"] = self.data[1:3,:]/mass
#        f["/pressure"] = (self.data[3,:]/vol - 0.5*(mass/vol)*(vx**2 + vy**2))*(self.gamma-1.0)

        dens = self.fields.get_field("density")
        velx = self.fields.get_field("velocity-x")
        vely = self.fields.get_field("velocity-y")
        pres = self.fields.get_field("pressure")

        f["/particles"]  = self.particles

        f["/density"]    = dens
        f["/velocity-x"] = velx
        f["/velocity-y"] = vely
        f["/pressure"]   = pres

        f.attrs["time"] = self.time

        f.close()



    def set_boundary_condition(self, boundary):

        if isinstance(boundary, BoundaryBase):
            self.boundary = boundary
        else:
            raise TypeError

    def set_reconstruction(self, reconstruction):

        if isinstance(reconstruction, ReconstructBase):
            self.reconstruction = reconstruction
        else:
            raise TypeError

    def set_initial_state(self, initial_particles, initial_data, initial_particles_index):
        """
        Set the initial state of the system by specifying the particle positions, their data
        U=(density, density*velocity, Energy) and particle labels (ghost or real).

        Parameters
        ----------
        initial_particles : Numpy array of size (dimensino, number particles)
        initial_data : Numpy array of conservative state vector U=(density, density*velocity, Energy)
            with size (variables, number particles)
        initial_particles_index: dictionary with two keys "real" and "ghost" that hold the indices
            in integer numpy arrays of real and ghost particles in the initial_particles array.
        """
        self.particles = initial_particles.copy()
        self.particles_index = dict(initial_particles_index)

        # make initial tesellation
        self.neighbor_graph, self.neighbor_graph_sizes, self.face_graph, self.face_graph_sizes, self.voronoi_vertices = self.mesh.tessellate(self.particles)

        # calculate volume of real particles 
        self.cell_info = self.mesh.volume_center_mass(self.particles, self.neighbor_graph, self.neighbor_graph_sizes, self.face_graph,
                self.voronoi_vertices, self.particles_index)

        # convert data to mass, momentum, and total energy in cell
        #self.data = initial_data*self.cell_info["volume"]

        num_particles = self.particles_index["real"].size

        # setup data container
        self.fields = Fields(num_particles, self.gamma, self.boundary)
        self.fields.create_fields()

        mass = self.fields.get_field("mass")
        momx = self.fields.get_field("momentum-x")
        momy = self.fields.get_field("momentum-y")
        ener = self.fields.get_field("energy")

        vol = self.cell_info["volume"]

        mass[:] = initial_data[0,:] * vol
        momx[:] = initial_data[1,:] * vol
        momy[:] = initial_data[2,:] * vol
        ener[:] = initial_data[3,:] * vol


    def set_riemann_solver(self, riemann_solver):

        if isinstance(riemann_solver, RiemannBase):
            self.riemann_solver = riemann_solver
        else:
            raise TypeError("Unknown riemann solver")

    def set_parameter(self, parameter_name, parameter):

        if parameter_name in self.__dict__.keys():
            setattr(self, parameter_name, parameter)
        else:
            raise ValueError("Unknown parameter: %s" % parameter_name)

    def solve(self):
        """
        Evolve the simulation from time zero to the specified max time.
        """
        #time = 0.0
        num_steps = 0

        while self.time < self.max_time and num_steps < self.max_steps:


            self.time += self._solve_one_step(num_steps)
            print "solving for step:", num_steps, "time: ", self.time


            # output data
            if num_steps%self.output_cycle == 0:
                self.data_dump(num_steps)

            num_steps+=1

#            # debugging plot --- turn to a routine later ---
#            l = []
#            ii = 0; jj = 0
#            for ip in self.particles_index["real"]:
#
#                jj += self.neighbor_graph_sizes[ip]*2
#                verts_indices = np.unique(self.face_graph[ii:jj])
#                verts = self.voronoi_vertices[verts_indices]
#
#                # coordinates of neighbors relative to particle p
#                xc = verts[:,0] - self.particles[0,ip]
#                yc = verts[:,1] - self.particles[1,ip]
#
#                # sort in counter clock wise order
#                sorted_vertices = np.argsort(np.angle(xc+1j*yc))
#                verts = verts[sorted_vertices]
#
#                l.append(Polygon(verts, True))
#
#                ii = jj
#
#
#            cells = self.particles_index["real"]
#
#            dens = self.fields.get_field("density")[cells]
#            velx = self.fields.get_field("velocity-x")[cells]
#            vely = self.fields.get_field("velocity-y")[cells]
#            pres = self.fields.get_field("pressure")[cells]
#
#            # add colormap
#            colors = []
#            for i in self.particles_index["real"]:
#                colors.append(dens[i])
#
#            #fig, ax = plt.subplots(figsize=(20, 5))
#            fig, ax = plt.subplots()
#            p = PatchCollection(l, alpha=0.4)
#            p.set_array(np.array(colors))
#            p.set_clim([0, 4])
#
#            ax.set_xlim(0,2)
#            ax.set_ylim(0,0.2)
#            ax.set_aspect(2)
#            ax.add_collection(p)
#
#            plt.colorbar(p, orientation='horizontal')
#            plt.savefig(self.output_name+`num_steps`.zfill(4))
#            plt.clf()
#
#
#
#            plt.figure(figsize=(5,5))
#            plt.subplot(3,1,1)
#            plt.scatter(self.particles[0,cells], dens, facecolors="none", edgecolors="r")
#            #plt.xlim(-0.2,2.2)
#            plt.ylim(-5,40)
#
#            plt.subplot(3,1,2)
#            plt.scatter(self.particles[0,cells], velx, facecolors="none", edgecolors="r")
#            #plt.xlim(-0.2,2.2)
#            plt.ylim(-5,30)
#
#            plt.subplot(3,1,3)
#
#            plt.scatter(self.particles[0,cells], pres, facecolors="none", edgecolors="r")
#            #plt.xlim(-0.2,2.2)
#            plt.ylim(-1,2000)
#
#            plt.savefig("scatter"+`num_steps`.zfill(4))
#            plt.clf()

        # last data dump
        self.data_dump(num_steps)


    def _solve_one_step(self, count):
        """
        Evolve the simulation for one time step.
        """

        # generate ghost particles with links to original real particles 
        self.particles = self.boundary.update(self.particles, self.particles_index, self.neighbor_graph, self.neighbor_graph_sizes)

        # make tesselation 
        self.neighbor_graph, self.neighbor_graph_sizes, self.face_graph, self.face_graph_sizes, self.voronoi_vertices = self.mesh.tessellate(self.particles)

        # calculate volume and center of mass of real particles
        self.cell_info = self.mesh.volume_center_mass(self.particles, self.neighbor_graph, self.neighbor_graph_sizes, self.face_graph,
                self.voronoi_vertices, self.particles_index)

        # calculate primitive variables of real particles
        self.fields.update_primitive(self.cell_info["volume"], self.particles, self.particles_index)
        #primitive = self._cons_to_prim(self.cell_info["volume"])

        # calculate global time step from real particles
        #dt = self.get_dt(self.time, primitive, self.cell_info["volume"])
        dt = self.get_dt()

        # assign primitive values to ghost particles
        #primitive = self.boundary.primitive_to_ghost(self.particles, primitive, self.particles_index)

        # assign particle velocities to real and ghost and do mesh regularization
        #w = self.mesh.assign_particle_velocities(self.particles, primitive, self.particles_index, self.cell_info, self.gamma, self.regularization)
        w = self.mesh.assign_particle_velocities(self.particles, self.fields.prim, self.particles_index, self.cell_info, self.gamma, self.regularization)

        # grab left and right faces
        left_face, right_face, faces_info = self.mesh.faces_for_flux(self.particles, self.fields.prim, w, self.particles_index, self.neighbor_graph,
                self.neighbor_graph_sizes, self.face_graph, self.voronoi_vertices)

#--->
        # calculate gradient for real particles
        gradx, grady = self.reconstruction.gradient(self.fields.prim, self.particles, self.particles_index, self.cell_info, self.neighbor_graph, self.neighbor_graph_sizes,
                self.face_graph, self.voronoi_vertices)

        # assign gradients to ghost particles
        gradx, grady = self.boundary.gradient_to_ghost(self.particles, gradx, grady, self.particles_index)


        # hack for right now
        ghost_map = self.particles_index["ghost_map"]
        cell_com = np.hstack((self.cell_info["center of mass"], self.cell_info["center of mass"][:, np.asarray([ghost_map[i] for i in self.particles_index["ghost"]])]))

        # find state values at the face 
        self.reconstruction.extrapolate(left_face, right_face, gradx, grady, faces_info, cell_com, self.gamma, dt)

#--->
        # calculate state at face by riemann solver
        fluxes = self.riemann_solver.fluxes(left_face, right_face, faces_info, self.gamma)

        # update conserved variables
        self.update(fluxes, dt, faces_info)

        # move particles
        self.particles[:,self.particles_index["real"]] += dt*w[:, self.particles_index["real"]]

        return dt


    def update(self, fluxes, dt, faces_info):

        ghost_map = self.particles_index["ghost_map"]
        area = faces_info["face areas"]

        k = 0
        for i, j in zip(faces_info["face pairs"][0,:], faces_info["face pairs"][1,:]):

            #self.data[:,i] -= dt*area[k]*fluxes[:,k]
            self.fields.field_data[:,i] -= dt*area[k]*fluxes[:,k]

            # do not update ghost particle cells
            if not ghost_map.has_key(j):
                #self.data[:,j] += dt*area[k]*fluxes[:,k]
                self.fields.field_data[:,j] += dt*area[k]*fluxes[:,k]

            k += 1
