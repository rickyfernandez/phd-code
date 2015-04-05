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

The particles are then redistributed by the load balance scheme
"""
import mpi4py.MPI as MPI
import numpy as np

from load_balance.load_balance import LoadBalance
from particles.particle_container import ParticleContainer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if size != 4:
    if rank == 0:
        raise RuntimeError("Run this test with 4 processors")

# create the initial distribution
if rank == 0:
    num_particles = 12
    x = np.array( [0.0, 1.0, 2.0, 3.0, 4.0,
                   0.0, 1.0, 2.0, 3.0, 4.0,
                   0.0, 1.0] , dtype=np.float64)

    y = np.array( [0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, 1.0, 1.0, 1.0, 1.0,
                   2.0, 2.0] , dtype=np.float64)

    gid = np.array( [0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10,
                     11], dtype=np.int8 )

if rank == 1:
    num_particles = 6
    x = np.array( [2.0, 3.0, 4.0,
                   0.0, 1.0, 2.0] , dtype=np.float64)

    y = np.array( [2.0, 2.0, 2.0,
                   3.0, 3.0, 3.0] , dtype=np.float64)

    gid = np.array( [12, 13, 14, 15,
                     16, 17], dtype=np.int8 )

if rank == 2:
    num_particles = 3
    x = np.array( [4.0, 3.0, 0.0] , dtype=np.float64)
    y = np.array( [3.0, 3.0, 4.0] , dtype=np.float64)

    gid = np.array( [18, 19, 20], dtype=np.int8 )


if rank == 3:
    num_particles = 4
    x = np.array( [1.0, 2.0, 3.0, 4.0] , dtype=np.float64)
    y = np.array( [4.0, 4.0, 4.0, 4.0] , dtype=np.float64)

    gid = np.array( [21, 22, 23, 24], dtype=np.int8 )


# create particle data structure
pc = ParticleContainer(num_particles)
pc['position-x'][:] = x
pc['position-y'][:] = y
pc['tag'][:] = gid

# Gather the global data on root
X = np.zeros(shape=25, dtype=np.float64)
Y = np.zeros(shape=25, dtype=np.float64)
GID = np.zeros(shape=25, dtype=np.int8)

displacements = np.array([12, 6, 3, 4], dtype=np.int32)

comm.Gatherv(sendbuf=x,  recvbuf=[X,  (displacements, None)], root=0)
comm.Gatherv(sendbuf=y,  recvbuf=[Y,  (displacements, None)], root=0)
comm.Gatherv(sendbuf=gid,recvbuf=[GID,(displacements, None)], root=0)

# brodcast global X, Y and GID to everyone
comm.Bcast(buf=X,  root=0)
comm.Bcast(buf=Y,  root=0)
comm.Bcast(buf=GID,root=0)

# perform the load decomposition
order = 3
load_b = LoadBalance(pc, comm, order=order)
load_b.decomposition()

# the particles are placed in a [0,4]x[0,4] box
for i in range(2):
    assert(load_b.global_xmin[i] == 0.0)
    assert(load_b.global_xmax[i] == 4.0)

# the box width is 4 times a fudge factor
assert(np.abs(4*1.001 - load_b.length) < 1.0E-10)

# make sure every key has been accounted for
assert(load_b.keys.size == num_particles)

# make sure hilbert keys are valid 
dim = 2
total_keys = 1 << (order*dim)
for i in range(num_particles):
    assert(load_b.keys[i] >= 0 and load_b.keys[i] < total_keys)

# make sure that all particles are accounted in building the
# global tree
assert(load_b.global_num_particles == 25)

# the work is just the number of particles in each leaf
# so the sum of each leaf should be the total number of particles
assert(np.sum(load_b.global_work) == 25)

for i in xrange(pc.num_particles):
    assert(abs(X[pc['tag'][i]] - pc['position-x'][i]) < 1e-15)
    assert(abs(Y[pc['tag'][i]] - pc['position-y'][i]) < 1e-15)
    assert(GID[pc['tag'][i]] == pc['tag'][i])
