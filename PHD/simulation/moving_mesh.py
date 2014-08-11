import h5py
import numpy as np
from PHD.mesh import voronoi_mesh
from PHD.reconstruction.reconstruction_base import reconstruction_base
from PHD.riemann.riemann_base import riemann_base
from PHD.boundary.boundary_base import boundary_base

# for debug plotting 
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib


class moving_mesh(object):

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
        self.data = None
        self.cell_info = None

        self.particles_index = None
        self.voronoi_vertices = None
        self.neighbor_graph = None
        self.neighbor_graph_sizes = None
        self.face_graph = None
        self.face_graph_sizes = None

        # simulation classes
        self.mesh = voronoi_mesh(regularization)
        self.boundary = None
        self.reconstruction = None
        self.riemann_solver = None


    def _get_dt(self, time, prim, vol):
        """
        Calculate the time step using the CFL condition.
        """
        # sound speed
        c = np.sqrt(self.gamma*prim[3,:]/prim[0,:])

        # calculate approx radius of each voronoi cell
        R = np.sqrt(vol/np.pi)

        dt = self.CFL*np.min(R/c)

        if time + dt > self.max_time:
            dt = self.max_time - time

        return dt


    def _cons_to_prim(self, volume):
        """
        Convert volume integrated variables (density, densiy*velocity, Energy) to
        primitive variables (mass, momentum, pressure).
        """
        # conserative vector is mass, momentum, total energy in cell volume
        mass = self.data[0,:]

        primitive = np.empty(self.data.shape, dtype=np.float64)
        primitive[0,:] = self.data[0,:]/volume      # density
        primitive[1:3,:] = self.data[1:3,:]/mass    # velocity

        # pressure
        primitive[3,:] = (self.data[3,:]/volume-0.5*primitive[0,:]*\
                (primitive[1,:]**2 + primitive[2,:]**2))*(self.gamma-1.0)

        return primitive

    def data_dump(self, time, num):

        f = h5py.File(self.output_name + "_" + `num`.zfill(4) + ".hdf5", "w")

        vol  = self.cell_info["volume"]
        mass = self.data[0,:]
        vx   = self.data[1,:]/mass
        vy   = self.data[2,:]/mass

        f["/particles"] = self.particles
        f["/density"]  = mass/vol
        f["/velocity"] = self.data[1:3,:]/mass
        f["/pressure"] = (self.data[3,:]/vol - 0.5*(mass/vol)*(vx**2 + vy**2))*(self.gamma-1.0)

        f.attrs["time"] = time

        f.close()



    def set_boundary_condition(self, boundary):

        if isinstance(boundary, boundary_base):
            self.boundary = boundary
        else:
            raise TypeError

    def set_reconstruction(self, reconstruction):

        if isinstance(reconstruction, reconstruction_base):
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
        self.data = initial_data*self.cell_info["volume"]


    def set_riemann_solver(self, riemann_solver):

        if isinstance(riemann_solver, riemann_base):
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
        time = 0.0
        num_steps = 0

        while time < self.max_time and num_steps < self.max_steps:


            time += self._solve_one_step(time, num_steps)
            print "solving for step:", num_steps, "time: ", time

            num_steps+=1

            # output data
            if num_steps%self.output_cycle == 0:
                self.data_dump(time, num_steps)

            # debugging plot --- turn to a routine later ---
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
#
#            # add colormap
#            colors = []
#            for i in self.particles_index["real"]:
#                colors.append(self.data[0,i]/self.cell_info["volume"][i])
#
#            #fig, ax = plt.subplots(figsize=(20, 5))
#            fig, ax = plt.subplots()
#            p = PatchCollection(l, alpha=0.4)
#            p.set_array(np.array(colors))
#            p.set_clim([0, 4.])
#
#            ax.set_xlim(0,1)
#            ax.set_ylim(0,1)
#            #ax.set_aspect(2)
#            ax.add_collection(p)
#
#            #plt.colorbar(p, orientation='horizontal')
#            plt.savefig(self.output_name+`num_steps`.zfill(4))
#            plt.clf()

            plt.figure(figsize=(5,5))
            plt.subplot(3,1,1)
            plt.scatter(self.particles[0, self.particles_index["real"]], self.data[0,:]/self.cell_info["volume"],
                    facecolors="none", edgecolors="r")
            plt.xlim(-0.2,2.2)
            #plt.ylim(-0.1,1.1)

            plt.subplot(3,1,2)
            plt.scatter(self.particles[0, self.particles_index["real"]], self.data[1,:], facecolors="none", edgecolors="r")
            plt.xlim(-0.2,2.2)
            plt.ylim(-2.1,2.1)

            plt.subplot(3,1,3)
            vol = self.cell_info["volume"]
            mass = self.data[0,:]
            vx = self.data[1,:]/mass
            vy = self.data[2,:]/mass

            p = (self.data[3,:]/vol - 0.5*(mass/vol)*(vx**2 + vy**2))*(self.gamma-1.0)
            plt.scatter(self.particles[0, self.particles_index["real"]], p, facecolors="none", edgecolors="r")
            plt.xlim(-0.2,2.2)
            #plt.ylim(-0.1,0.5)

            plt.savefig("scatter"+`num_steps`.zfill(4))
            plt.clf()



    def _solve_one_step(self, time, count):
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
        primitive = self._cons_to_prim(self.cell_info["volume"])

        # calculate global time step from real particles
        dt = self._get_dt(time, primitive, self.cell_info["volume"])

        # assign primitive values to ghost particles
        primitive = self.boundary.primitive_to_ghost(self.particles, primitive, self.particles_index)

        # assign particle velocities to real and ghost and do mesh regularization
        w = self.mesh.assign_particle_velocities(self.particles, primitive, self.particles_index, self.cell_info, self.gamma)

        # grab left and right faces
        left_face, right_face, faces_info = self.mesh.faces_for_flux(self.particles, primitive, w, self.particles_index, self.neighbor_graph,
                self.neighbor_graph_sizes, self.face_graph, self.voronoi_vertices)

        # transform to rest frame ? boost to face ?
        self.mesh.transform_to_face(left_face, right_face, faces_info)
#2
#--->
        # calculate gradient for real particles
        gradx, grady = self.reconstruction.gradient(primitive, self.particles, self.particles_index, self.cell_info, self.neighbor_graph, self.neighbor_graph_sizes,
                self.face_graph, self.voronoi_vertices)

        # assign gradients to ghost particles
        gradx, grady = self.boundary.gradient_to_ghost(self.particles, gradx, grady, self.particles_index)


        # hack for right now
        ghost_map = self.particles_index["ghost_map"]
        cell_com = np.hstack((self.cell_info["center of mass"], self.cell_info["center of mass"][:, np.asarray([ghost_map[i] for i in self.particles_index["ghost"]])]))

        # find state values at the face 
        self.reconstruction.extrapolate(left_face, right_face, gradx, grady, faces_info, cell_com, self.gamma, dt)
#2
#--->

        # rotate to frame 
        self.mesh.rotate_to_face(left_face, right_face, faces_info)

        # calculate state at face by riemann solver
        face_states = self.riemann_solver.state(left_face, right_face, self.gamma)

        # transform back to lab frame
        fluxes = self.mesh.transform_to_lab(face_states, faces_info)

        # update conserved variables
        self._update(fluxes, dt, faces_info)

        # move particles
        self.particles[:,self.particles_index["real"]] += dt*w[:, self.particles_index["real"]]

        return dt


    def _update(self, fluxes, dt, faces_info):

        ghost_map = self.particles_index["ghost_map"]
        #area = faces_info[1,:]
        area = faces_info["face areas"]

        k = 0
        #for i, j in zip(faces_info[4,:], faces_info[5,:]):
        for i, j in zip(faces_info["face pairs"][0,:], faces_info["face pairs"][1,:]):

            self.data[:,i] -= dt*area[k]*fluxes[:,k]

            # do not update ghost particle cells
            if not ghost_map.has_key(j):
                self.data[:,j] += dt*area[k]*fluxes[:,k]

            k += 1
