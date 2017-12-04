import numpy as np
from ..utils.tools import check_class

class SimulationTime(object):
    '''
    Controller of signaling the simulation when to
    data output and finish
    '''
    def __init__(self):
        self.outputs  = set()
        self.finishes = set()

    #@check_class(SimulationOutputer)
    def add_output(self, output):
        '''Add output criteria to set'''
        self.outputs.add(output)

    #@check_class(SimulationFinisher)
    def add_finish(self, finish):
        '''Add finish criteria criteria to set'''
        self.finishes.add(finish)

    def finished(self, integrator):
        '''
        Cycle through all finishers and check for flag to end
        simulation

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
            finish_sim = finish.finished(integrator) or finish_sim
        return finish_sim

    def output(self, output_directory, integrator):
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
        for output in self.outputs:
            output.output(output_directory, integrator)

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
            dt = min(dt, finish.modify_timestep(integrator))
        for output in self.outputs:
            dt = min(dt, output.modify_timestep(integrator))
        return dt
