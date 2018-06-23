# containers
from phd.containers.containers import \
        CarrayContainer

# geometry
from phd.mesh.mesh import \
        Mesh

from phd.domain.boundary import \
        BoundaryConditionBase, \
        Reflective, \
        Periodic

from phd.domain.domain_manager import \
        DomainManager

from phd.load_balance.load_balance import \
        LoadBalance

# physics
from phd.equation_state.equation_state import \
        EquationStateBase, \
        IdealGas

from phd.gravity.gravity_tree import \
        GravityTree

from phd.gravity.interaction import \
        GravityAcceleration

from phd.gravity.splitter import \
        BarnesHut

# computation
from phd.simulation.simulation import \
        Simulation

from phd.integrate.integrate import \
        IntegrateBase, \
        StaticMeshMUSCLHancock, \
        MovingMeshMUSCLHancock, \
        Nbody
        #MovingMeshPakmor

from phd.reconstruction.reconstruction import \
        ReconstructionBase, \
        PieceWiseConstant, \
        PieceWiseLinear

from phd.riemann.riemann import \
        RiemannBase, \
        HLL, \
        HLLC, \
        Exact

from phd.gravity.gravity_force import \
        ConstantGravity, \
        SelfGravity

# input and output
from phd.io.simulation_time_manager import \
        SimulationTimeManager
        #Iteration, \
        #IterationInterval, \
        #Time, \
        #TimeInterval, \
        #SelectedTimes

from phd.io.simulation_finish import \
        SimulationFinisherBase, \
        Iteration, \
        Time

from phd.io.simulation_output import \
        SimulationOutputterBase, \
        IterationInterval, \
        TimeInterval, \
        InitialOutput, \
        FinalOutput

from phd.io.read_write import \
        ReaderWriterBase, \
        Hdf5

# helpers
from phd.utils.particle_tags import \
        ParticleTAGS

from phd.utils.plot_voro import \
        vor_collection

#from phd.utils.store_class import \
#        to_dict

#from phd.utils.tools import \
#        create_components_timeshot


from phd.utils.particle_creator import \
        HydroParticleCreator

from phd.utils.logger import \
        phdLogger

from phd.utils.parallelize import \
        distribute_initial_particles

try:
    import mpi4py.MPI as MPI
    _has_mpi = True
except ImportError:
    _has_mpi = False

if _has_mpi:
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
    _in_parallel = _comm.Get_size() > 1
else:
    _comm = None
    _rank = 0
    _size = 1
    _in_parallel = False
