import h5py
import numpy as np
from ..utils.logger import phdLogger
from ..utils.particle_tags import ParticleTAGS

class ReaderWriter(object):
    #def __init__(self, folder_name="data", pad=4):
    def __init__(self):
        #self.folder_name = folder_name
        #self.pad = pad
        pass

    def write(self, base_name, output_directory, integrator):
        pass

    def read(self):
        pass

class Hdf5(ReaderWriter):
    def write(self, base_name, output_directory, integrator):
        '''
        Write particle data to hdf5 file
        '''
        file_name = output_directory + '/' + base_name + '.hdf5'
        #file_name = output_directory + '/' + self.folder_name
        #file_name += '_' + str(integrator.iteration).zfill(self.pad) + '.hdf5'

        phdLogger.info(
                "hdf5 format: Writting %s at "
                "iteration %d, time %f, dt %f" %\
                (file_name,
                 integrator.iteration,
                 integrator.time,
                 integrator.dt))

        with h5py.File(file_name, 'w') as f:

            # store current time
            f.attrs["dt"] = integrator.dt
            f.attrs["time"] = integrator.time
            f.attrs["iteration"] = integrator.iteration

            # store particle data
            particle_grp = f.create_group('particles')

            # common information 
            particle_grp.attrs['Real'] = ParticleTAGS.Real
            particle_grp.attrs['Ghost'] = ParticleTAGS.Real
            particle_grp.attrs['number_particles'] = integrator.particles.get_number_of_items()

            # store particle data for each field
            for prop_name in integrator.particles.properties.keys():
                data_grp = particle_grp.create_group(prop_name)
                data_grp.attrs['dtype'] = integrator.particles.carray_info[prop_name]
                data_grp.create_dataset('data', data=integrator.particles[prop_name])

            f.close()

        phdLogger.success("hdf5 format: %s write success" %\
                file_name)

    def read(self, file_name):
        '''
        Read hdf5 file of particles

        Parameters
        ----------
        file_name : str
            File name to be read in
        '''
        phdLogger.info("hdf5 format: Reading filename %s" %\
                file_name)

        with h5py.File(file_name, 'r') as f:

            particle_grp = f['particles']
            num_particles = particles_grp.attrs['number_particles']
            particles = CarrayContainer(num_particles)

            # populate arrays with data
            for field in particle_grp.keys():
                field_grp = particle_grp[field]
                particles.register(num_particles, field, field_grp['dtype'])
                particles[field][:] = field_grp['data'][:]

        phdLogger.sucess("hdf5 format: particles created successfully from %s" %\
                file_name)

        return particles



# --- scratch
#            # store named groups
#            named_grp = f.create_group('named_group')
#            for grp, grp_list in pc.named_groups.iteritems():
#                named_grp.attrs[grp] = ','.join(grp_list)
#            particle_grp = f['named_group']
#            for grp in named_grp.keys():
#                .named_gropus[grp] = named_grp[grp].split(',')
