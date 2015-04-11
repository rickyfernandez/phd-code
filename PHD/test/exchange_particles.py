"""Test the ParticleArrayExchange object with the following data

25 particles are created on a rectangular grid with the following
processor assignment:

2---- 3---- 3---- 3---- 3
|     |     |     |     |
1---- 1---- 1---- 2---- 2
|     |     |     |     |
0---- 0---- 1---- 1---- 1
|     |     |     |     |
0---- 0---- 0---- 0---- 0
|     |     |     |     |
0---- 0---- 0---- 0---- 0


We assume that after a load balancing step, we have the following
distribution:

1---- 1---- 1---- 3---- 3
|     |     |     |     |
1---- 1---- 1---- 3---- 3
|     |     |     |     |
0---- 0---- 0---- 3---- 3
|     |     |     |     |
0---- 0---- 2---- 2---- 2
|     |     |     |     |
0---- 0---- 2---- 2---- 2

We create a ParticleArray to represent the initial distribution and
use ParticleArrayExchange to move the data by manually setting the
particle import/export lists. We require that the test be run with 4
processors.

"""
import mpi4py.MPI as MPI
import numpy as np

from particles.particle_container import ParticleContainer
from load_balance.load_balance import LoadBalance


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 4:
    if rank == 0:
        raise RuntimeError("Run this test with 4 processors")

# create the initial distribution
if rank == 0:

    num_particles = 12

    pc = ParticleContainer(num_particles)
    pc['position-x'][:] = np.array( [0.0, 1.0, 2.0, 3.0, 4.0,
                                     0.0, 1.0, 2.0, 3.0, 4.0,
                                     0.0, 1.0] , dtype=np.float64)

    pc['position-y'][:] = np.array( [0.0, 0.0, 0.0, 0.0, 0.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0,
                                     2.0, 2.0] , dtype=np.float64)

    # tags are not ment for global id but for this test it will
    pc['tag'][:] = np.array([0, 1, 2, 3, 4, 5,
                             6, 7, 8, 9, 10, 11], dtype=np.int8)

    numExport = 6
    exportLocalids = np.array( [2, 3, 4, 7, 8, 9], dtype=np.int32 )
    exportProcs = np.array( [2, 2, 2, 2, 2, 2], dtype=np.int32 )

if rank == 1:

    num_particles = 6

    pc = ParticleContainer(num_particles)
    pc['position-x'][:] = np.array( [2.0, 3.0, 4.0,
                                     0.0, 1.0, 2.0] , dtype=np.float64)

    pc['position-y'][:] = np.array( [2.0, 2.0, 2.0,
                                     3.0, 3.0, 3.0] , dtype=np.float64)

    pc['tag'][:] = np.array([12, 13, 14, 15,
                             16, 17], dtype=np.int8)

    numExport = 3
    exportLocalids = np.array( [0, 1, 2], dtype=np.int32 )
    exportProcs = np.array( [0, 3, 3], dtype=np.int32 )

if rank == 2:

    num_particles = 3

    pc = ParticleContainer(num_particles)
    pc['position-x'][:] = np.array( [4.0, 3.0, 0.0] , dtype=np.float64)
    pc['position-y'][:] = np.array( [3.0, 3.0, 4.0] , dtype=np.float64)

    pc['tag'][:] = np.array([18, 19, 20], dtype=np.int8)

    numExport = 3
    exportLocalids = np.array( [0, 1, 2], dtype=np.int32 )
    exportProcs = np.array( [3, 3, 1], dtype=np.int32 )

if rank == 3:

    num_particles = 4

    pc = ParticleContainer(num_particles)
    pc['position-x'][:] = np.array( [1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    pc['position-y'][:] = np.array( [4.0, 4.0, 4.0, 4.0], dtype=np.float64)

    pc['tag'][:] = np.array([21, 22, 23, 24], dtype=np.int8)

    numExport = 2
    exportLocalids = np.array( [0,1], dtype=np.int32 )
    exportProcs = np.array( [1, 1], dtype=np.int32 )


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Gather the global data on root
X = np.zeros(shape=25, dtype=np.float64)
Y = np.zeros(shape=25, dtype=np.float64)
GID = np.zeros(shape=25, dtype=np.int8)

displacements = np.array([12, 6, 3, 4], dtype=np.int32)

comm.Gatherv(sendbuf=pc['position-x'], recvbuf=[X, (displacements, None)],   root=0)
comm.Gatherv(sendbuf=pc['position-y'], recvbuf=[Y, (displacements, None)],   root=0)
comm.Gatherv(sendbuf=pc['tag'],        recvbuf=[GID, (displacements, None)], root=0)

# brodcast global X, Y and GID to everyone
comm.Bcast(buf=X, root=0)
comm.Bcast(buf=Y, root=0)
comm.Bcast(buf=GID, root=0)

# use the load balance to exchange the particles
lb = LoadBalance(pc, factor=1.0, comm=comm)
lb.export_proc = exportProcs
lb.export_ids  = exportLocalids
lb.exchange_particles()

# after the exchange each proc should have 6 particles except for proc 0
numParticles = 6
if rank == 0:
    numParticles = 7

assert(pc.num_real_particles == numParticles)

for i in xrange(pc.num_real_particles):
    assert(abs(X[pc['tag'][i]] - pc['position-x'][i]) < 1e-15)
    assert(abs(Y[pc['tag'][i]] - pc['position-y'][i]) < 1e-15)
    assert(GID[pc['tag'][i]] == pc['tag'][i])
