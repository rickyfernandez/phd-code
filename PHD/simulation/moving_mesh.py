import numpy as np
from PHD.mesh import voronoi_mesh
from PHD.riemann.riemann_base import riemann_base
from PHD.boundary.boundary_base import boundary_base

from PHD.mesh.cell_volume_center import number_of_faces

# for debug plotting 
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib


class moving_mesh(object):

    def __init__(self, gamma = 1.4, CFL = 0.5, max_steps=100, max_time=None,
            output_name="simulation_"):

        # simulation parameters
        self.CFL = CFL
        self.gamma = gamma
        self.max_steps = max_steps
        self.max_time = max_time
        self.output_name = output_name

        # particle information
        self.particles = None
        self.data = None
        self.cell_info = None

        self.particles_index = None
        self.voronoi_vertices = None
        self.ng = None
        self.ngs = None
        self.fg = None
        self.fgs = None
#-->
        self.neighbor_graph = None
        self.face_graph = None
#-->

        # simulation classes
        self.mesh = voronoi_mesh()
        self.boundary = None
        self.reconstruction = None
        self.riemann_solver = None


    def _get_dt(self, prim, vol):
        """
        Calculate the time step using the CFL condition.
        """
        # sound speed
        c = np.sqrt(self.gamma*prim[3,:]/prim[0,:])

        # calculate approx radius of each voronoi cell
        R = np.sqrt(vol/np.pi)

        return self.CFL*np.min(R/c)


    def _cons_to_prim(self, volume):
        """
        Convert volume integrated variables (density, densiy*velocity, Energy) to
        primitive variables (mass, momentum, pressure).
        """
        # conserative vector is mass, momentum, total energy
        mass = self.data[0,:]

        primitive = np.empty(self.data.shape, dtype=np.float64)
        primitive[0,:] = self.data[0,:]/volume      # density
        primitive[1:3,:] = self.data[1:3,:]/mass    # velocity

        # pressure
        primitive[3,:] = (self.data[3,:]/volume-0.5*self.data[0,:]*\
                (primitive[1,:]**2 + primitive[2,:]**2))*(self.gamma-1.0)

        return primitive



    def set_boundary_condition(self, boundary):

        if isinstance(boundary, boundary_base):
            self.boundary = boundary
        else:
            raise TypeError

    def set_initial_state(self, initial_particles, initial_data, initial_particles_index):
        """
        Set the initial state of the system by specifying the particle positions, their data
        U=(density, density*velocity, Energy) and particle labels (ghost or real).

        Parameters
        ----------
        initial_particles : Numpy array of size (number particles, dimension)
        initial_data : Numpy array of conservative state vector U=(density, density*velocity, Energy)
            with size (variables, number particles)
        initial_particles_index: dictionary with two keys "real" and "ghost" that hold the indices
            in integer numpy arrays of real and ghost particles in the initial_particles array.
        """
        self.particles = initial_particles.copy()
        self.particles_index = dict(initial_particles_index)

        # make initial tesellation
        self.neighbor_graph, self.face_graph, self.voronoi_vertices, self.ng, self.ngs, self.fg, self.fgs = self.mesh.tessellate(self.particles)

        # calculate volume of real particles 
        self.cell_info = self.mesh.volume_center_mass(self.particles, self.ng, self.ngs, self.fg, self.voronoi_vertices, self.particles_index)

        # convert data to mass, momentum, and energy
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

            print "solving for step:", num_steps

            time += self._solve_one_step(time, num_steps)


            # debugging plot --- turn to a routine later ---
            l = []
            for i in self.particles_index["real"]:

                verts_indices = np.unique(np.asarray(self.face_graph[i]).flatten())
                verts = self.voronoi_vertices[verts_indices]

                # coordinates of neighbors relative to particle p
                xc = verts[:,0] - self.particles[i,0]
                yc = verts[:,1] - self.particles[i,1]

                # sort in counter clock wise order
                sorted_vertices = np.argsort(np.angle(xc+1j*yc))
                verts = verts[sorted_vertices]

                l.append(Polygon(verts, True))

            # add colormap
            colors = []
            for i in self.particles_index["real"]:
                #colors.append(self.data[0,i]/self.cell_info[0,i])
                colors.append(self.data[0,i]/self.cell_info["volume"][i])

            fig, ax = plt.subplots()
            p = PatchCollection(l, alpha=0.4)
            p.set_array(np.array(colors))
            p.set_clim([0, 4.])
            ax.add_collection(p)
            plt.colorbar(p)
            plt.savefig(self.output_name+`num_steps`.zfill(4))
            plt.clf()

            num_steps+=1



    def _solve_one_step(self, time, count):
        """
        Evolve the simulation for one time step.
        """

        # generate periodic ghost particles with links to original real particles 
        self.particles = self.boundary.update(self.particles, self.particles_index, self.ng, self.ngs)

        # make tesselation returning graph of neighbors graph of faces and voronoi vertices
        self.neighbor_graph, self.face_graph, self.voronoi_vertices, self.ng, self.ngs, self.fg, self.fgs = self.mesh.tessellate(self.particles)

        self.cell_info = self.mesh.volume_center_mass(self.particles, self.ng, self.ngs, self.fg, self.voronoi_vertices, self.particles_index)

        volume = self.cell_info["volume"]

        # calculate primitive variables for real particles
        primitive = self._cons_to_prim(volume)

        # calculate global time step from real particles
        dt = self._get_dt(primitive, volume)
        if time + dt > self.max_time:
            dt = self.max_time - time

        # copy values for ghost particles
        ghost_map = self.particles_index["ghost_map"]
        primitive = np.hstack((primitive,
            primitive[:, np.asarray([ghost_map[i] for i in self.particles_index["ghost"]])]))

        # reverse velocities
        self.boundary.reverse_velocities(self.particles, primitive, self.particles_index)

        # mesh regularization
        w = self.mesh.regularization(primitive, self.particles, self.gamma, self.cell_info, self.particles_index)
        w = np.zeros(w.shape)
        w = np.hstack((w, w[:, np.asarray([ghost_map[i] for i in self.particles_index["ghost"]])]))

        # add particle velocities
        w[:, self.particles_index["real"]]  += primitive[1:3, self.particles_index["real"]]
        w[:, self.particles_index["ghost"]] += primitive[1:3, self.particles_index["ghost"]]

        # grab each face with particle id of the left and right particles as well angle and area
        #faces_info = self.mesh.faces_for_flux(self.particles, w, self.particles_index, self.neighbor_graph,
        #        self.face_graph, self.voronoi_vertices)

        faces_info = self.mesh.faces_for_flux2(self.particles, w, self.particles_index, self.ng, self.ngs,
                self.fg, self.voronoi_vertices)

        #if count == 2:
        #    import pdb; pdb.set_trace()

        # grab left and right states
        left  = primitive[:, faces_info[4,:].astype(int)]
        right = primitive[:, faces_info[5,:].astype(int)]

        # calculate state at edges
        fluxes = self.riemann_solver.flux(left, right, faces_info, self.gamma)

        #numFaces = number_of_faces(self.ng, self.ngs, self.particles_index["real"].size)
        #print "number of faces", fluxes.shape, numFaces

        #assert(fluxes.shape[1] == number_of_faces(self.ng, self.ngs, self.particles_index["real"].size))

        # update conserved variables
        self._update(fluxes, dt, faces_info)

        # move particles
        self.particles[self.particles_index["real"],:] += dt*np.transpose(w[:, self.particles_index["real"]])

        return dt


    def _update(self, fluxes, dt, face_info):

        ghost_map = self.particles_index["ghost_map"]
        area = face_info[1,:]

        k = 0
        for i, j in zip(face_info[4,:], face_info[5,:]):

            self.data[:,i] -= dt*area[k]*fluxes[:,k]

            # do not update ghost particle cells
            if not ghost_map.has_key(j):
                self.data[:,j] += dt*area[k]*fluxes[:,k]

            k += 1
