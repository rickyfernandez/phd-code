from ..utils.tools import check_class
from .simulation_finish import SimulationFinisherBase
from .simulation_output import SimulationOutputterBase

class SimulationTimeManager(object):
    """Collection of data outputters and simulation finishers.

    Controls when the simulation needs to output data or needs to end the simulation.

    Attributes
    ----------
    _outputs : set
        Set of SimulationOutputter to signal the simualtion
        to output data to disk.

    _finishes : set
        Set of SimulationFinisher to signal the termination
        of the simulation.

    """
    def __init__(self, **kwargs):
        self._outputs  = set()
        self._finishes = set()

    @check_class(SimulationOutputterBase)
    def add_output(self, output):
        """Add output criteria to set."""
        self._outputs.add(output)

    @check_class(SimulationFinisherBase)
    def add_finish(self, finish):
        """Add finish criteria criteria to set"""
        self._finishes.add(finish)

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
        for finish in self._finishes:
            finish_sim = finish.finished(integrator) or finish_sim
        return finish_sim

    def output(self, simulation):
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
        for output in self._outputs:
            output.output(simulation)

    def modify_timestep(self, simulation):
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
        dt = simulation.integrator.dt
        for finish in self._finishes:
            dt = min(dt, finish.modify_timestep(simulation))
        for output in self._outputs:
            dt = min(dt, output.modify_timestep(simulation))
        return dt
