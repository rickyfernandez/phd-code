import phd
import numpy as np

def create_particles(gamma):

    Lx = 1.    # domain size in x
    nx = 100   # particles per dim
    n = nx*nx  # number of points

    rho_1 = 1.0; rho_2 = 2.0
    vel = 0.5; amp = 0.05

    dx = Lx/nx # spacing between particles

    # create particle container
    pc = phd.ParticleContainer(n)
    part = 0
    for i in range(nx):
        for j in range(nx):

            x = (i+0.5)*dx
            y = (j+0.5)*dx

            pert = amp*np.sin(4.*np.pi*x)

            if 0.25 < y and y < 0.75:

                pc['density'][part] = rho_1
                pc['velocity-x'][part] = -(vel + pert)

            else:

                pc['density'][part] = rho_2
                pc['velocity-x'][part] = vel + pert


            pc['position-x'][part] = x
            pc['position-y'][part] = y
            pc['velocity-y'][part] = pert
            pc['ids'][part] = part
            part += 1

    pc['pressure'][:] = 2.5
    pc['velocity-y'][:] = 0.0
    pc['tag'][:] = phd.ParticleTAGS.Real
    pc['type'][:] = phd.ParticleTAGS.Undefined

    return pc

# create inital state of the simulation
pc = create_particles(1.4)

domain = phd.DomainLimits(dim=2, xmin=0., xmax=1.)           # spatial size of problem 
boundary = phd.Boundary(domain,                              # periodic boundary condition
        boundary_type=phd.BoundaryType.Periodic)
mesh = phd.Mesh(boundary)                                    # tesselation algorithm
reconstruction = phd.PieceWiseConstant()                     # constant reconstruction
riemann = phd.HLLC(reconstruction, gamma=1.4)                 # riemann solver
integrator = phd.MovingMesh(pc, mesh, riemann, regularize=1) # integrator 
solver = phd.Solver(integrator,                              # simulation driver
        cfl=0.5, tf=2.5, pfreq=25,
        relax_num_iterations=0,
        output_relax=False,
        fname='kh_cartesian')
solver.solve()
