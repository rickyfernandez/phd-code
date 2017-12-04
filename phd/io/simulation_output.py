
class SimulationOutputer(object):
    '''
    Class that singals the simulation to write out
    data at current time.
    '''
    def __init__(self):
        self.read_write = None
        self.counter = 0

    def set_writer(self, read_write):
        self.read_write = read_write

    def output(self, output_directory, integrator):
        if self.check_for_output(integrator):
            self.read_write.write(output_directory, integrator)

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
        return (integrator.iteration % self.iteration_interval == 0)

#class TimeInterval(SimulationOutputer):
#    def __init__(self, time_interval, time_last_output=0):
#        self.time_interval = time_interval
#        self.time_last_output = time_last_output
#
#    def output(self, integrator):
#        '''
#        Return True to signal the simulation has reached
#        multiple of time_interval to ouput data
#
#        Parameters
#        ----------
#        integrator : phd.IntegrateBase
#            Integrator that solves the equations
#
#        Returns
#        -------
#        bool
#            True if should output False otherwise
#        '''
#        if integrator.time >= self.time_last_output + self.time_interval:
#            self.time_last_output += self.time_interval
#            self.read_write.write()
#
#class SelectedTimes(SimulationOutputer):
#    def __init__(self, output_times):
#        self.output_times = np.asarray(output_times)
#
#        self.remaining = self.output_times.size
#        self.times_not_done = np.ones(
#                self.remaining,
#                dtype=bool)
#
#    def output(self):
#        '''
#        Return True to signal the simulation has reached
#        selected time to ouptput data
#
#        Parameters
#        ----------
#        integrator : phd.IntegrateBase
#            Integrator that solves the equations
#
#        Returns
#        -------
#        bool
#            True if should output False otherwise
#        '''
#        if self.remaining:
#            flag = integrator.time >= self.output_times[self.times_not_done]
#
#            self.times_not_done[flag] = False
#            self.remaining = self.times_not_done.sum()
#            self.read_write.write()
#
#    def modify_timestep(self, integrate):
#        '''
#        Parameters
#        ----------
#        integrator : phd.IntegrateBase
#            Integrator that solves the equations
#
#        Returns
#        -------
#        float
#            modified time step if needed otherwise integrator dt
#        '''
#        if self.remaining:
#            dt =  self.output_times[self.times_not_done] - integrator.time
#            return np.min(dt.min(), integrator.dt)
#        else:
#            return integrate.dt
