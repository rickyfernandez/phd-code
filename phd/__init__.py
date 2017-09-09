from phd.boundary.boundary import \
        BoundaryType, Boundary, \
        BoundaryParallel

from phd.containers.containers import \
        CarrayContainer

from phd.domain.domain import \
        DomainLimits

# -------- hack delete later --------------
from phd.integrate.new_integrator import \
        NewIntegrateBase

from phd.integrate.integrator import \
        IntegrateBase, \
        MovingMesh

from phd.load_balance.load_balance import \
        LoadBalance

from phd.gravity.gravity_tree import \
        GravityTree

from phd.gravity.interaction import \
        GravityAcceleration

from phd.gravity.splitter import \
        BarnesHut

from phd.mesh.mesh import \
        Mesh

from phd.reconstruction.reconstruction import \
        ReconstructionBase, \
        PieceWiseConstant, \
        PieceWiseLinear

from phd.riemann.riemann import \
        RiemannBase, \
        HLL, \
        HLLC, \
        Exact

from phd.simulation.simulation import \
        Simulation

from phd.utils.particle_tags import \
        ParticleTAGS

from phd.utils.plot_voro import \
        vor_collection

from phd.utils.store_class import \
        to_dict

from phd.utils.particle_creator import \
        HydroParticleCreator

from phd.utils.logger import \
        phdLogger

from phd.io.simulation_time import \
        SimulationTime, \
        Iteration, \
        IterationInterval, \
        Time, \
        TimeInterval, \
        SelectedTimes

try:
    import mpi4py.MPI as mpi
    _has_mpi = True
except ImportError:
    _has_mpi = False

if _has_mpi:
    _comm = mpi.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
    _in_parallel = _comm.Get_size() > 1
else:
    _comm = None
    _rank = 0
    _size = 1
    _in_parallel = False
