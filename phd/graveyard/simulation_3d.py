import PHD.riemann as riemann
import PHD.boundary as boundary
import PHD.simulation as simulation
import PHD.reconstruction as reconstruction
import PHD.test_problems.three_dim.toro_1 as toro_1

# parameters and initial state of the simulation
parameters, data, particles, particles_index = toro_1.simulation()

# create boundary and riemann objects
boundary_condition = boundary.Reflect3D(0.,1.,0.,.1,0.,.1)
#reconstruction = reconstruction.PiecewiseConstant3D(boundary_condition)
reconstruction = reconstruction.PiecewiseLinear3D(boundary_condition)
#riemann_solver = riemann.Exact3D(reconstruction)
#riemann_solver = riemann.Hllc3D(reconstruction)
riemann_solver = riemann.Hll3D(reconstruction)

# setup the moving mesh simulation
simulation = simulation.MovingMesh3D()
#simulation = simulation.StaticMesh3D()

for key, value in parameters.iteritems():
    simulation.set_parameter(key, value)

# set the boundary, riemann solver, and initial state of the simulation 
simulation.set_boundary_condition(boundary_condition)
simulation.set_reconstruction(reconstruction)
simulation.set_riemann_solver(riemann_solver)
simulation.set_initial_state(particles, data, particles_index)

# run the simulation
simulation.solve()
