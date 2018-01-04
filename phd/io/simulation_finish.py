
class SimulationFinisherBase(object):
    """Class that singals the simulation to stop evolving.

    Attributes
    ----------
    """
    def finished(self, simulation):
        """Check for flag to end the simulation.
        Parameters
        ----------
        simulation : Simulation
           Class that marshalls the simulation of the fluid equations.

        Returns
        -------
        bool
           True if main loop should be exited 
        """
        msg = "SimulationFinisher::finish called!"
        raise NotImplementedError(msg)

    def modify_timestep(self, simulation):
        """Return consistent time step.

        Parameters
        ----------
        simulation : Simulation
           Class that marshalls the simulation of the fluid equations.

        Returns
        -------
        float
            Modified dt if needed otherwise integrator dt.

        """
        return simulation.integrator.dt

class Iteration(SimulationFinisherBase):
    """Signal simulation complete if reached desired iteration
    number.

    Attributes
    ----------
    iteration_max : int
        Iteration value to end the simulation.

    """
    def __init__(self, iteration_max):
        self.iteration_max = iteration_max

    def finished(self, simulation):
        """Return True to signal the simulation is finished
        if reached max iteration number.
        """
        if simulation.integrator.iteration == self.iteration_max:
            return True
        else:
            return False

class Time(SimulationFinisherBase):
    """Signal simulation complete if reached desired time.

    Attributes
    ----------
    time_max : float
        Time to end the simulation.

    """
    def __init__(self, time_max):
        self.time_max = time_max

    def finished(self, simulation):
        """Return True to signal the simulation is finished
        if the simulation reached a certain time.
        """
        if simulation.integrator.time >= self.time_max:
            return True
        else:
            return False

    def modify_timestep(self, integrator):
        """Check if the simulation has reached final time."""
        integrator = simulation.integrator
        if integrator.time + integrator.dt >= self.time_max:
            return self.time_max - integrator.time
        else:
            return integrator.dt
