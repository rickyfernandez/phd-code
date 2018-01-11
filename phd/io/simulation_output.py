import os

from ..utils.logger import phdLogger

class SimulationOutputterBase(object):
    """Class that signals the simulation to write out
    data at current state of the simultion.

    This class is the api for all outputters. To inherit
    you will have write your own check_for_output and
    modify_timestep methods.

    Attributes
    ----------
    base_name : str
        Name of file to be outputted. Name will be post
        fixed with output number.

    counter : int
        Output number in output order.

    output_directory : str
        Directory to store all output of the simulation.

    read_write : ReaderWriter
        Class that writes data in a given format.

    pad : int
        Number of padding spaces for output numbering of
        data files.

    """
    def __init__(self, base_name, counter=0, pad=4):
        self.pad = pad
        self.counter = counter
        self.base_name = base_name

        # defined by setters
        self.read_write = None
        self.output_directory = None

    def initialize(self):
        """Initialize outputter, check for ReaderWriter and output directory."""
        if not self.read_write or not self.output_directory:
            raise RuntimeError("ERROR: Not all setters defined in %s!" %\
                    self.__class__.__name__)

        # directory to store data
        self.data_directory = self.output_directory + "/" + self.base_name

        if os.path.isdir(self.data_directory):
            phdLogger.warning("Directory %s already exists, "
                    "files maybe over written!" % self.data_directory)
        else:
            os.mkdir(self.data_directory)

    def set_output_directory(self, output_directory):
        """Set directory where to output files."""
        self.output_directory = output_directory

    def set_writer(self, read_write):
        """Set class to write out data in a specific format.

        Parameters
        ----------
        read_write : ReaderWriter
            Class that writes data in a given format.

        """
        self.read_write = read_write

    def check_for_output(self, simulation):
        """Signal if simulation needs to output.

        Parameters
        ----------
        simulation : Simulation
           Class that marshalls the simulation of the fluid equations.

        Returns
        -------
        bool
            True if time to output.

        """
        msg = "SimulationOutputterBase::check_for_output called!"
        raise NotImplementedError(msg)

    def create_directory(self):

        # new directory
        output_directory = self.data_directory + "/" +\
                self.base_name + str(self.counter).zfill(self.pad)

        if os.path.isdir(output_directory):
            phdLogger.warning("Directory %s already exists, "
                    "files maybe over written!" % output_directory)
        else:
            os.mkdir(output_directory)

    def output(self, simulation):
        """Write out data to given directory.

        Parameters
        ----------
        simulation : Simulation
           Class that marshalls the simulation of the fluid equations.

        """
        if self.check_for_output(simulation):
            self.create_directory()

            output_directory = self.data_directory + "/" +\
                self.base_name + str(self.counter).zfill(self.pad)
            file_name = self.base_name + str(self.counter).zfill(self.pad)

            self.read_write.write(
                    file_name, output_directory, simulation.integrator)

            # create json file
            self.write_parameters(file_name, self.output_directory, simulation)
            self.counter += 1

    def modify_timestep(self, simulation):
        """
        Return consistent time step.

        Parameters
        ----------
        simulation : Simulation
           Class that marshalls the simulation of the fluid equations.

        Returns
        -------
        float
            Modified dt if needed otherwise integrator dt.

        """
        msg = "SimulationOutputterBase::modify_timestep called!"
        raise NotImplementedError(msg)

    def write_parameters(self, file_name, output_directory, simulation):
        pass
        # save parameter information
        #param = save_parameters(simulation)
        #with open(self.base_name + str(self.counter).zfill(self.pad), "w") as fp:
        #    json.dump(param, fp, sort_keys=True, indent=4)


class IterationInterval(SimulationOutputterBase):
    """Class that signals the simulation to write out
    at given iteration intervals.

    Attributes
    ----------
    base_name : str
        Name of file to be outputted. Name will be post
        fixed with output number.

    counter : int
        Output number in output order.

    iteration_interval : int
        Number of iterations between outputs.

    pad : int
        Number of padding spaces for output numbering of
        data files.

    read_write : ReaderWriter
        Class that writes data in a given format.

    """
    def __init__(self, iteration_interval, base_name="iteration_interval",
                 counter=0, pad=4):
        super(IterationInterval, self).__init__(base_name, counter, pad)
        self.iteration_interval = iteration_interval

    def check_for_output(self, simulation):
        """Return True to signal the simulation has reached
        iteteration interval to ouput data."""
        integrator = simulation.integrator
        return (integrator.iteration % self.iteration_interval == 0)

    def modify_timestep(self, simulation):
        """Return consistent time step."""
        # not modifying
        return simulation.integrator.dt


class InitialOutput(SimulationOutputterBase):
    """Class that signals the simulation to write out
    initial data.

    Attributes
    ----------
    base_name : str
        Name of file to be outputted. Name will be post
        fixed with output number.

    counter : int
        Output number in output order.

    pad : int
        Number of padding spaces for output numbering of
        data files.

    read_write : ReaderWriter
        Class that writes data in a given format.

    """
    def __init__(self, base_name="initial_output",
                 counter=0, pad=4):
        super(InitialOutput, self).__init__(base_name, counter, pad)

    def check_for_output(self, simulation):
        """Return True to signal first output before main loop."""
        return simulation.state == SimulationTAGS.BEFORE_LOOP

    def output(self, simulation):
        """Write out data to given directory."""
        if self.check_for_output(simulation):
            file_name = self.base_name
            self.read_write.write(
                    file_name, self.output_directory, simulation.integrator)

            # create json file
            self.write_parameters(file_name, self.output_directory, simulation)

    def modify_timestep(self, simulation):
        """Return consistent time step."""
        # not modifying
        return simulation.integrator.dt


class FinalOutput(SimulationOutputterBase):
    """Class that signals the simulation to write out
    final data.

    Attributes
    ----------
    base_name : str
        Name of file to be outputted. Name will be post
        fixed with output number.

    counter : int
        Output number in output order.

    pad : int
        Number of padding spaces for output numbering of
        data files.

    read_write : ReaderWriter
        Class that writes data in a given format.

    """
    def __init__(self, base_name="final_output",
                 counter=0, pad=4):
        super(InitialOutput, self).__init__(base_name, counter, pad)

    def check_for_output(self, simulation):
        """Return True to signal final output after main loop."""
        return simulation.state == SimulationTAGS.AFTER_LOOP

    def output(self, simulation):
        """Write out data to given directory."""
        if self.check_for_output(simulation):
            file_name = self.base_name
            self.read_write.write(
                    file_name, self.output_directory, simulation.integrator)

            # create json file
            self.write_parameters(file_name, self.output_directory, simulation)

    def modify_timestep(self, simulation):
        """Return consistent time step."""
        # not modifying
        return simulation.integrator.dt


class TimeInterval(SimulationOutputterBase):
    """Class that signals the simulation to write out
    final data.

    Attributes
    ----------
    base_name : str
        Name of file to be outputted. Name will be post
        fixed with output number.

    counter : int
        Output number in output order.

    pad : int
        Number of padding spaces for output numbering of
        data files.

    next_time_output : float
        Time of next output.

    read_write : ReaderWriter
        Class that writes data in a given format.

    time_interval : float
        Time between outputs.

    time_last_output : float
        Time of last output.

    """
    def __init__(self, time_interval, initial_time=0,
                 base_name="time_interval", counter=0, pad=4):
        super(InitialOutput, self).__init__(base_name, counter, pad)

        self.time_interval = time_interval
        self.time_last_output = initail_time
        self.next_time_output = time_last_output + time_interval

    def initialize(self):
        if not self.read_write:
            raise RuntimeError("ERROR: Data writer not defined!")

    def check_for_output(self, simulation):
        """Return True to signal the simulation has reached
        multiple of time_interval to ouput data.
        """
        integrator = simulation.integrator
        return (integrator.iteration % self.iteration_interval == 0)

        integrator = simulation.integrator
        if integrator.time >= self.next_time_output:
            self.time_last_output = integrator.time
            self.next_time_output = self.time_last_output + self.time_interval
            return True

    def modify_timestep(self, simulation):
        """Return time step for next time interval output."""
        return self.next_time_output - simulation.integrator.time
