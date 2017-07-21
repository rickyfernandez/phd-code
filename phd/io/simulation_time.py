import numpy as np

from ..utils.tools import check_class

class SimulationTime(object):
    '''
    Controller of signaling the simulation when to
    data output and finish
    '''
    def __int__(self):
        self.outputs  = set()
        self.finishes = set()

    @check_class(SimulationOutputer)
    def add_output(output):
        '''Add output criteria to set'''
        self.outputs.add(output)

    @check_class(SimulationFinisher)
    def add_finish(finish):
        '''Add finish criteria criteria to set'''
        self.finishes.add(finish)

    def finish(self, integrator):
        '''
        Cycle through all finishers and check for flag to end
        the simulation

        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        bool
            True if simulation finished False otherwise
        '''
        finish_sim = False
        for finish in self.finishes:
            finish_sim = finish.finish(integrator) or finish_sim
        return finish_sim

    def output(self, integrator):
        '''
        Cycle through all outputs and check for flag to write out
        simulation data

        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        bool
            True if should output False otherwise
        '''
        output_sim = False
        for output in self.outputers:
            output_sim = output.output(integrator) or output_sim
        return output_sim

    def modify_timestep(self, integrator):
        '''
        Return the smallest time step from each finish and output
        object in the simulation

        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        float
            modified time step if needed otherwise integrator dt
        '''
        dt = integrator.dt
        for finish in self.finishes:
            dt = min(dt, finish.modify_timestep(integrate))
        for output in self.outputs:
            dt = min(dt, output.modify_timestep(integrate))
        return dt

class SimulationFinisher(object):
    '''
    Class that singals the simulation to stop evolving
    '''
    def finish(self, integrator):
        '''
        Check for flag to end the simulation
        '''
        msg = "SimulationFinisher::finish called!"
        raise NotImplementedError(msg)

    def modify_timestep(self, integrator):
        '''
        Return consistent time step

        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        float
            modified time step if needed otherwise integrator dt
        '''
        return integrator.dt

class SimulationOutputer(object):
    '''
    Class that singals the simulation to write out
    data at current time.
    '''
    def output(self):
        '''
        Check for flag to write outsimulation data
        '''
        msg = "SimulationOutputer::output called!"
        raise NotImplementedError(msg)

    def modify_timestep(self, integrator):
        '''
        Return consistent time step

        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        float
            modified time step if needed otherwise integrator dt
        '''
        return integrator.dt

class Iteration(SimulationFinisher):
    '''
    Signal simulation complete if reached desired iteration
    number
    '''
    def __init__(self, iteration_max):
        self.iteration_max = iteration_max

    def finished(self, integrator):
        '''
        Return True to signal the simulation is finished
        if the simulation reached max iteration number

        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        bool
            True if simulation finished False otherwise
        '''
        if integrator.iteration == self.iteration_max:
            return True
        else:
            return False

class IterationInterval(SimulationOutputer):
    def __init__(self, iteration_interval):
        self.iteration_interval = iteration_interval

    def check_for_output(self, integrator):
        '''
        Return True to signal the simulation has reached
        iteteration number to ouput data

        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        bool
            True if should output False otherwise
        '''
        if integrator.iteration % self.iteration_interval == 0:
            return True
        else:
            return False

class Time(SimulationFinisher):
    def __init__(self, time_max):
        self.time_max = time_max

    def finished(self, integrator):
        '''
        Return True to signal the simulation is finished
        if the simulation reached a certain time

        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        bool
            True if simulation finished False otherwise
        '''
        if self.time_max >= integrator.time:
            return True
        else:
            return False

    def modify_timestep(self, integrator):
        '''
        Check if the simulation has reached final time.

        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        float
            modified time step if needed otherwise integrator dt
        '''
        if integrator.time + integrator.dt >= self.time_max:
            return self.time_max - integrator.time
        else:
            return integrator.dt

class TimeInterval(SimulationOutputer):
    def __init__(self, time_interval, time_last_output=0):
        self.time_interval = time_interval
        self.time_last_output = time_last_output

    def output(self, integrator):
        '''
        Return True to signal the simulation has reached
        multiple of time_interval to ouput data

        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        bool
            True if should output False otherwise
        '''
        if integrator.time >= self.time_last_output + self.time_interval:
            self.time_last_output += self.time_interval
            return True
        else:
            return False

class SelectedTimes(SimulationOutputer):
    def __init__(self, output_times):
        self.output_times = np.asarray(output_times)

        self.remaining = self.output_times.size
        self.times_not_done = np.ones(
                self.remaining,
                dtype=bool)

    def output(self):
        '''
        Return True to signal the simulation has reached
        selected time to ouptput data

        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        bool
            True if should output False otherwise
        '''
        if self.remaining:
            flag = integrator.time >= self.output_times[self.times_not_done]

            self.times_not_done[flag] = False
            self.remaining = self.times_not_done.sum()
            return True
        else:
            return False

    def modify_timestep(self, integrate):
        '''
        Parameters
        ----------
        integrator : phd.IntegrateBase
            Integrator that solves the equations

        Returns
        -------
        float
            modified time step if needed otherwise integrator dt
        '''
        if self.remaining:
            dt =  self.output_times[self.times_not_done] - integrator.time
            return np.min(dt.min(), integrator.dt)
        else:
            return integrate.dt
