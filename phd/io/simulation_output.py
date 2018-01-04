
class SimulationOutputerBase(object):
    """Class that singals the simulation to write out
    data at current state of the simultion.

    This class is the api for all outputters. To inherit
    you will have write your done, check_for_output,
    done, and modify_timestep methods.

    Attributes
    ----------
    base_name : str
        Name of file to be outputted. Name will be post
        fixed with output number.

    counter : int
        Output number in output order.

    read_write : ReaderWriter 
        Class that writes data in a given format.

    pad : int
        Number of padding spaces for output numbering of
        data files.

    """
    def __init__(self, base_name, counter=0, pad=4):
        self.pad = pad
        self.counter = counter
        self.read_write = None
        self.base_name = base_name 

    def initialize(self, simulation):
        if not self.read_write:
            raise RuntimeError("ERROR: Data writer not defined!")

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
        msg = "SimulationOutputerBase::check_for_output called!"
        raise NotImplementedError(msg)

    def done(self, simulation):
        """Signal if outputter is done outputting.

        Parameters
        ----------
        simulation : Simulation
           Class that marshalls the simulation of the fluid equations.

        Returns
        -------
        bool
            True if no more outputs to perform.

        """
        msg = "SimulationOutputerBase::done called!"
        raise NotImplementedError(msg)

    def output(self, output_directory, simulation):
        """Write out data to given directory.

        Parameters
        ----------
        output_directory : str
            Directory name to output data.

        simulation : Simulation
           Class that marshalls the simulation of the fluid equations.

        """
        # skip if outputs are done
        if self.done(simulation):
            return

        if self.check_for_output(simulation):
            file_name = self.base_name + str(self.counter).zfill(self.pad)
            self.read_write.write(
                    file_name, output_directory, simulation.integrator)
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
        msg = "SimulationOutputerBase::modify_timestep called!"
        raise NotImplementedError(msg)

class IterationInterval(SimulationOutputerBase):
    """Class that singals the simulation to write out
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
    def __init__(self, iteration_interval, base_name="iteration_interval_",
                 counter=0, pad=4):
        super(IterationInterval, self).__init__(base_name, counter, pad)
        self.iteration_interval = iteration_interval

    def done(self):
        """Signal if outputter is done outputting."""
        return False

    def check_for_output(self, simulation):
        """Return True to signal the simulation has reached
        iteteration interval to ouput data."""
        integrator = simulation.integrator
        return (integrator.iteration % self.iteration_interval == 0)

    def modify_timestep(self, simulation):
        """Return consistent time step."""
        # not modifying
        return simulation.integrator.dt

class InitialOutput(SimulationOutputerBase):
    """Class that singals the simulation to write out
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

    simulation_started : bool
        True if simaultion has entered compuation loop

    """
    def __init__(self, base_name="initial_output",
                 counter=0, pad=4):
        super(InitialOutput, self).__init__(base_name, counter, pad)
        self.simulation_started = False

    def done(self, simulation):
        """Signal if outputter has already outputted initial data."""
        if not self.simulation_started:
            if simulation.state == SimulationTAGS.BEFORE_LOOP:
                self.simulation_started = True
            return False
        return True

    def check_for_output(self, simulation):
        """Return True to signal first output before main loop."""
        return simulation.state == SimulationTAGS.BEFORE_LOOP

    def output(self, output_directory, simulation):
        """Write out data to given directory."""
        # skip if outputs are done
        if self.done():
            return

        if self.check_for_output(simulation):
            file_name = self.base_name
            self.read_write.write(
                    file_name, output_directory, simulation.integrator)
            self.counter += 1

    def modify_timestep(self, simulation):
        """Return consistent time step."""
        # not modifying
        return simulation.integrator.dt


class FinalOutput(SimulationOutputerBase):
    """Class that singals the simulation to write out
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

    simulation_finished : bool
        True if simaultion has exited compuation loop

    """
    def __init__(self, base_name="final_output",
                 counter=0, pad=4):
        super(InitialOutput, self).__init__(base_name, counter, pad)
        self.simulation_finished = False

    def done(self, simulation):
        """Signal if outputter has already outputted final data."""
        return False

    def check_for_output(self, simulation):
        """Return True to signal final output after main loop."""
        return simulation.state == SimulationTAGS.AFTER_LOOP

    def output(self, output_directory, simulation):
        """Write out data to given directory."""
        # skip if outputs are done
        if self.done():
            return

        if self.check_for_output(simulation):
            file_name = self.base_name
            self.read_write.write(
                    file_name, output_directory, simulation.integrator)
            self.counter += 1

    def modify_timestep(self, simulation):
        """Return consistent time step."""
        # not modifying
        return simulation.integrator.dt


class TimeInterval(SimulationOutputer):
    """Class that singals the simulation to write out
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
                 base_name, counter, pad):
        super(InitialOutput, self).__init__(base_name, counter, pad)

        self.time_interval = time_interval
        self.time_last_output = initail_time
        self.next_time_output = time_last_output + time_interval

    def initialize(self):
        if not self.read_write:
            raise RuntimeError("ERROR: Data writer not defined!")

    def done(self, simulation):
        """Signal if outputter has already outputted time intervals."""
        return False

    def output(self, simulation):
        """Return True to signal the simulation has reached
        multiple of time_interval to ouput data.
        """
        integrator = simulation.integrator
        if integrator.time >= self.time_last_output + self.time_interval:
            self.time_last_output = integrator.time 
            self.next_time_output = self.time_last_output + self.time_interval
            self.read_write.write()

    def modify_timestep(self, simulation):
        """Return time step for next time interval output."""
        return self.next_time_output - simulation.integrator.time


class SelectedTimes(SimulationOutputer):
    """Class that singals the simulation to write out
    data at selected times.

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

    output_times : list
        List of times to output at.

    read_write : ReaderWriter 
        Class that writes data in a given format.

    times_not_done : np.array
        Boolean array that signals which output times are left.

    """
    def __init__(self, output_times):

        # TODO should remove output times instead of bool array
        if type(output_times) is not list:
            raise RuntimeError("ERROR: output_times needs to be type list")

        self.output_times = np.asarray(output_times)
        self.times_not_done = np.ones(
                self.output_times.size,
                dtype=bool)

    def done(self, simulation):
        """Signal if outputter has already outputted time intervals."""
        return self.times_not_done.sum() == 0

    def output(self, simulation):
        """Return True to signal the simulation has reached
        selected time to ouptput data.
        """
        integrator = simulation.integrator
        if self.times_not_done.sum():
            flag = integrator.time >= self.output_times[self.times_not_done]

            self.times_not_done[flag] = False
            self.read_write.write()

    def modify_timestep(self, simulation):
        """Return time step needed for next selected time."""
        if not self.done():
            integrator = simulation.integrator
            dt = self.output_times[self.times_not_done] - integrator.time
            return np.min(dt.min(), integrator.dt)
        else:
            return simulation.integrate.dt
