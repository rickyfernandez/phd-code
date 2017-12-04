import numpy as np
from ..utils.tools import check_class

class SimulationFinisher(object):
    '''
    Class that singals the simulation to stop evolving
    '''
    def finished(self, integrator):
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
        if integrator.time >= self.time_max:
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
