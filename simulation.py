import PHD.riemann as riemann
import PHD.boundary as boundary
import PHD.simulation as simulation
import PHD.reconstruction as reconstruction
import PHD.test_problems.sedov as sedov

# parameters for the simulation
CFL = 0.3
gamma = 1.4
max_steps = 1000
max_time = 0.1
output_name = "Sod"
output_cycle = 1000
#regularization = True

# create boundary and riemann objects
boundary_condition = boundary.Reflect(0.,1.,0.,1.)
#reconstruction = reconstruction.PiecewiseConstant(boundary_condition)
reconstruction = reconstruction.PiecewiseLinear(boundary_condition)
#riemann_solver = riemann.Exact(reconstruction)
#riemann_solver = riemann.Hllc(reconstruction)
riemann_solver = riemann.Hll(reconstruction)

# create initial state of the system
data, particles, particles_index = sedov.simulation()

# setup the moving mesh simulation
simulation = simulation.MovingMesh()
#simulation = simulation.StaticMesh()

# set runtime parameters for the simulation
simulation.set_parameter("CFL", CFL)
simulation.set_parameter("gamma", gamma)
simulation.set_parameter("max_steps", max_steps)
simulation.set_parameter("max_time", max_time)
simulation.set_parameter("output_name", output_name)
simulation.set_parameter("output_cycle", output_cycle)
#simulation.set_parameter("regularization", regularization)

# set the boundary, riemann solver, and initial state of the simulation 
simulation.set_boundary_condition(boundary_condition)
simulation.set_reconstruction(reconstruction)
simulation.set_riemann_solver(riemann_solver)
simulation.set_initial_state(particles, data, particles_index)

# run the simulation
simulation.solve()
