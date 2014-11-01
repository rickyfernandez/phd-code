import h5py
import numpy as np
import simulation as sim
from PHD.fields import Fields3D
from PHD.mesh import VoronoiMesh3D
from static_mesh import StaticMesh


class MovingMesh3D(StaticMesh):
    """
    moving mesh simulation class
    """

    def __init__(self, gamma = 1.4, CFL = 0.5, max_steps=100, max_time=None, output_cycle = 100000,
            output_name="simulation_", regularization=True):

        # simulation parameters
        self.CFL = CFL
        self.gamma = gamma
        self.max_steps = max_steps
        self.max_time = max_time
        self.output_cycle = output_cycle
        self.output_name = output_name

        # particle information
        self.particles = None
        self.fields = None
        self.cells_info = None
        self.particles_index = None

        # particle graph information
        self.graphs = None

        # runtime parameters
        self.dt = 0.
        self.time = 0.
        self.num_steps = 0

        # simulation classes
        self.mesh = VoronoiMesh3D()
        self.boundary = None
        self.reconstruction = None
        self.riemann_solver = None

        # simulation parameters
        self.regularization = regularization


    def data_dump(self, num):

        f = h5py.File(self.output_name + "_" + `num`.zfill(4) + ".hdf5", "w")

        dens = self.fields.get_field("density")
        velx = self.fields.get_field("velocity-x")
        vely = self.fields.get_field("velocity-y")
        velz = self.fields.get_field("velocity-z")
        pres = self.fields.get_field("pressure")

        f["/particles"]  = self.particles

        f["/density"]    = dens
        f["/velocity-x"] = velx
        f["/velocity-y"] = vely
        f["/velocity-z"] = velz
        f["/pressure"]   = pres

        f.attrs["time"] = self.time

        f.close()

    def get_dt(self):
        """
        Calculate the time step using the CFL condition.
        """

        vol = self.cells_info["volume"]

        # moving mesh solvers have different courant restraint depending if solved
        # in lab or moving frame
        self.dt = self.CFL*self.riemann_solver.get_dt(self.fields, vol, self.gamma)

        # correct time step if exceed max time
        if self.time + self.dt > self.max_time:
            self.dt = self.max_time - self.time

    def set_initial_state(self, initial_particles, initial_data, initial_particles_index):

        self.particles = initial_particles.copy()
        self.particles_index = dict(initial_particles_index)

        # make initial tesellation
        self.graphs = self.mesh.tessellate(self.particles)

        # calculate volume of real particles 
        self.cells_info, _ = self.mesh.cell_and_faces_info(self.particles, self.particles_index, self.graphs)

        num_particles = self.particles_index["real"].size

        # setup data container
        self.fields = Fields3D(num_particles, self.gamma, self.boundary)
        self.fields.create_fields()

        mass = self.fields.get_field("mass")
        momx = self.fields.get_field("momentum-x")
        momy = self.fields.get_field("momentum-y")
        momz = self.fields.get_field("momentum-z")
        ener = self.fields.get_field("energy")

        vol = self.cells_info["volume"]

        mass[:] = initial_data[0,:] * vol
        momx[:] = initial_data[1,:] * vol
        momy[:] = initial_data[2,:] * vol
        momz[:] = initial_data[3,:] * vol
        ener[:] = initial_data[4,:] * vol


#    def solve(self):
#        """
#        Evolve the simulation from initial time to the specified max time.
#        """
#
#        while self.time < self.max_time and self.num_steps < self.max_steps:
#
#            print "solving for step:", self.num_steps, "time: ", self.time
#
#            # advance the solution for one time step
#            self.solve_one_step()
#
#            self.time += self.dt
#            self.num_steps += 1
#
#
#            # output data
#            if self.num_steps%self.output_cycle == 0:
#                self.data_dump(self.num_steps)


    def solve_one_step(self):
        """
        Evolve the simulation for one time step.
        """

        # generate ghost particles with links to original real particles 
        self.particles = self.fields.update_boundaries(self.particles, self.particles_index, self.graphs)

        # construct the new mesh 
        self.graphs = self.mesh.tessellate(self.particles)

        # output data
#        if self.num_steps%self.output_cycle == 0:
#            self.data_dump(self.num_steps)

        # calculate volume and center of mass of real particles
        self.cells_info, faces_info = self.mesh.cell_and_faces_info(self.particles, self.particles_index, self.graphs)

#        print "Total domain volume:", np.sum(self.cells_info["volume"])

        # calculate primitive variables of real particles and pass to ghost particles with give boundary conditions
        self.fields.update_primitive(self.cells_info["volume"], self.particles, self.particles_index)

        # calculate global time step
        self.get_dt()

        # assign fluid velocities to particles, regularize if needed, and pass to ghost particles
        w = self.mesh.assign_particle_velocities(self.particles, self.fields, self.particles_index, self.cells_info, self.gamma, self.regularization)

        # assign face velocities
        self.mesh.assign_face_velocities(self.particles, self.particles_index, self.graphs, faces_info, w)


        # calculate gradient for real particles and pass to ghost particles
        #self.reconstruction.gradient(self.fields.prim, self.particles, self.particles_index, self.cells_info, self.graphs)

        # extrapolate state to face, apply frame transformations, solve riemann solver, and transform back
        fluxes = self.riemann_solver.fluxes(self.particles, self.particles_index, self.graphs, self.fields.prim, self.cells_info,
                faces_info, self.gamma, self.dt)

        # update conserved variables
        self.update(self.fields, fluxes, faces_info)

        # update particle positions
        self.move_particles(w)


    def move_particles(self, w):
        """
        advance real particles positions for one time step
        """
        self.particles[:,self.particles_index["real"]] += self.dt*w[:, self.particles_index["real"]]
