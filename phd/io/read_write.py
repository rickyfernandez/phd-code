import h5py
import numpy as np
from ..utils.logger import phdLogger
from ..utils.particle_tags import ParticleTAGS
from ..containers.containers import CarrayContainer

class ReaderWriterBase(object):
    def write(self, base_name, output_directory, integrator):
        """Write simulation to disk.

        This base class is the api for writing and reading
        of data. Note the user is free to write any type of
        output. For example, this can be used to create
        runtime plots or diagnostics.

        Parameters
        ----------
        base_name : str
            File name for output data.

        output_directory : str
            Directory where to store the data.

        integrator : IntegrateBase
            Advances the fluid equations by one step.
        """
        msg = "ReaderWriterBase::write called!"
        raise NotImplementedError(msg)

    def read(self, file_name):
        """Read data and return particles.

        For formats where the data is being stored, a read
        function must be defined. This function reads the
        data outputted by `write` and returns a CarrayContainer
        of particles to be used by the simulation.

        Parameters
        ----------
        file_name : str
            File name to be read in.

        Returns
        -------
        particles : CarrayContainer
            Complete container of particles for a simulation.

        """
        msg = "ReaderWriterBase::read called!"
        raise NotImplementedError(msg)

class Hdf5(ReaderWriterBase):
    def write(self, base_name, output_directory, integrator):
        """Write simulation data to hdf5 file."""
        file_name = base_name + ".hdf5"
        output_path = output_directory + "/" + file_name
        phdLogger.info("hdf5 format: Writting %s" % file_name)

        with h5py.File(output_path, "w") as f:

            # store current time
            f.attrs["dt"] = integrator.dt
            f.attrs["time"] = integrator.time
            f.attrs["iteration"] = integrator.iteration

            # store particle data
            particle_grp = f.create_group("particles")

            # common information 
            particle_grp.attrs["Real"] = ParticleTAGS.Real
            particle_grp.attrs["Ghost"] = ParticleTAGS.Ghost
            particle_grp.attrs["number_particles"] = integrator.particles.get_carray_size()

            # store particle data for each field
            for prop_name in integrator.particles.carrays.keys():
                data_grp = particle_grp.create_group(prop_name)
                data_grp.attrs["dtype"] = integrator.particles.carray_dtypes[prop_name]
                data_grp.create_dataset("data", data=integrator.particles[prop_name])

            f.close()

    def read(self, file_name):
        """Read hdf5 file of particles."""
        phdLogger.info("hdf5 format: Reading filename %s" % file_name)

        with h5py.File(file_name, "r") as f:

            particle_grp = f["particles"]
            num_particles = particle_grp.attrs["number_particles"]
            particles = CarrayContainer(num_particles)

            # populate arrays with data
            for field_key in particle_grp.keys():
                field = field_key.encode('utf8')

                field_grp = particle_grp[field]
                particles.register_carray(num_particles, field, field_grp.attrs["dtype"])
                particles[field][:] = field_grp["data"][:]

        return particles
