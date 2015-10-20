import h5py
import numpy as np
import simulation as sim
from PHD.fields import Fields3D
from PHD.mesh import VoronoiMesh3D
from static_mesh import StaticMesh


class StaticMesh3D(StaticMesh):
    """
    moving mesh simulation class
    """

    def __init__(self, gamma = 1.4, CFL = 0.5, max_steps=100, max_time=None, output_cycle = 100000,
            output_name="simulation_"):

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

        # grab values that correspond to real particles
        dens = self.fields.get_field("density")
        velx = self.fields.get_field("velocity-x")
        vely = self.fields.get_field("velocity-y")
        velz = self.fields.get_field("velocity-z")
        pres = self.fields.get_field("pressure")

        # sound speed
        c = np.sqrt(self.gamma*pres/dens)

        # calculate approx radius of each voronoi cell
        R = (3.*vol/(4.*np.pi))**(1./3.)

        dt_x = R/(abs(velx) + c)
        dt_y = R/(abs(vely) + c)
        dt_z = R/(abs(velz) + c)

        self.dt = self.CFL*min(dt_x.min(), dt_y.min(), dt_z.min())

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
