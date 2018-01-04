import numpy as np
from ..utils.tools import check_class

class SimulationTime(object):
    """Collection of data outputters and simulation finishers.

    Controls when the simulation needs to output data or needs to end the simulation.

    Attributes
    ----------
    outputs : set
        Set of SimulationOutputer to signal the simualtion
        to output data to disk.

    finishes : set
        Set of SimulationFinisher to signal the termination
        of the simulation.
        
    """
    def __init__(self):
        self.outputs  = set()
        self.finishes = set()

    #@check_class(SimulationOutputer)
    def add_output(self, output):
        """Add output criteria to set."""
        self.outputs.add(output)

    #@check_class(SimulationFinisher)
    def add_finish(self, finish):
        """Add finish criteria criteria to set"""
        self.finishes.add(finish)

    def finished(self, integrator):
        """Cycle through all finishers and check for flag to end
        the simulation.

        Parameters
        ----------
        integrator : IntegrateBase
            Integrator that solves the equations.

        Returns
        -------
        bool
            True if simulation finished False otherwise.

        """
        finish_sim = False
        for finish in self.finishes:
            finish_sim = finish.finished(integrator) or finish_sim
        return finish_sim

    def output(self, output_directory, simulation):
        """Cycle through all outputs and check for flag to write out
        simulation data.

        Parameters
        ----------
        simulation : Simulation
           Class that marshalls the simulation of the fluid equations.

        Returns
        -------
        bool
            True if should output False otherwise.
        """
        for output in self.outputs:
            output.output(output_directory, simulation)

    def modify_timestep(self, integrator):
        """Return the smallest time step from each finish and output
        object in the simulation. Assumes that integrator has its
        dt updated.

        Parameters
        ----------
        integrator : IntegrateBase
            Integrator that solves the equations.

        Returns
        -------
        float
            modified time step if needed otherwise integrator dt.
        """
        dt = integrator.dt
        for finish in self.finishes:
            dt = min(dt, finish.modify_timestep(integrator))
        for output in self.outputs:
            dt = min(dt, output.modify_timestep(integrator))
        return dt
