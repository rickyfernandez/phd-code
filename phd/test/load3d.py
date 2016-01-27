"""Test the ParticleArrayExchange object with the following data

125 particles are created on a rectangular grid with the following
processor assignment along eeach slice along the z-axis:

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

from domain.domain import DomainLimits
from load_balance.load_balance import LoadBalance3D
from containers.containers import ParticleContainer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if size != 4:
    if rank == 0:
        raise RuntimeError("Run this test with 4 processors")

# create the initial distribution
if rank == 0:
    num_particles = 12*5
    x = np.array( 5*[0.0, 1.0, 2.0, 3.0, 4.0,
                   0.0, 1.0, 2.0, 3.0, 4.0,
                   0.0, 1.0], dtype=np.float64)

    y = np.array( 5*[0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, 1.0, 1.0, 1.0, 1.0,
                   2.0, 2.0], dtype=np.float64)

    z = np.repeat(np.arange(5), 12).astype(np.float64)

    gid = np.arange(60, dtype=np.int32)
    #gid = np.array( [0, 1, 2, 3, 4, 5,
    #                 6, 7, 8, 9, 10,
    #                 11], dtype=np.int32 )

if rank == 1:
    num_particles = 6*5
    x = np.array( 5*[2.0, 3.0, 4.0,
                   0.0, 1.0, 2.0] , dtype=np.float64)

    y = np.array( 5*[2.0, 2.0, 2.0,
                   3.0, 3.0, 3.0] , dtype=np.float64)

    z = np.repeat(np.arange(5), 6).astype(np.float64)

    gid = np.arange(60, 60 + num_particles, dtype=np.int32)
    #gid = np.array( [12, 13, 14, 15,
    #                 16, 17], dtype=np.int32 )

if rank == 2:
    num_particles = 3*5
    x = np.array( 5*[4.0, 3.0, 0.0], dtype=np.float64)
    y = np.array( 5*[3.0, 3.0, 4.0], dtype=np.float64)
    z = np.repeat(np.arange(5), 3).astype(np.float64)

    gid = np.arange(90, 90 + num_particles, dtype=np.int32)
    #gid = np.array( [18, 19, 20], dtype=np.int32 )


if rank == 3:
    num_particles = 4*5
    x = np.array( 5*[1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array( 5*[4.0, 4.0, 4.0, 4.0], dtype=np.float64)
    z = np.repeat(np.arange(5), 4).astype(np.float64)

    gid = np.arange(105, 105 + num_particles, dtype=np.int32)
    #gid = np.array( [21, 22, 23, 24], dtype=np.int32 )


# create particle data structure
pa = ParticleContainer(num_particles)
pa.register_property(num_particles, 'position-z', 'double')
pa['position-x'][:] = x
pa['position-y'][:] = y
pa['position-z'][:] = z

pa.register_property(x.size, 'gid', 'long')
pa['gid'][:] = gid

# Gather the global data on root
X   = np.zeros(shape=125, dtype=np.float64)
Y   = np.zeros(shape=125, dtype=np.float64)
Z   = np.zeros(shape=125, dtype=np.float64)
GID = np.zeros(shape=125, dtype=np.int32)

displacements = 5*np.array([12, 6, 3, 4], dtype=np.int32)

comm.Gatherv(sendbuf=x,  recvbuf=[X,  (displacements, None)], root=0)
comm.Gatherv(sendbuf=y,  recvbuf=[Y,  (displacements, None)], root=0)
comm.Gatherv(sendbuf=z,  recvbuf=[Z,  (displacements, None)], root=0)
comm.Gatherv(sendbuf=gid,recvbuf=[GID,(displacements, None)], root=0)

# brodcast global X, Y and GID to everyone
comm.Bcast(buf=X,   root=0)
comm.Bcast(buf=Y,   root=0)
comm.Bcast(buf=Z,   root=0)
comm.Bcast(buf=GID, root=0)

# perform the load decomposition
dom = DomainLimits(dim=3, xmin=0., xmax=4.)
order = 3
load_b = LoadBalance3D(pa, dom, comm=comm, factor=1.0, order=order)
load_b.decomposition()

# make sure every key has been accounted for
assert(load_b.keys.size == num_particles)

# make sure hilbert keys are valid 
dim = 3
total_keys = 1 << (order*dim)
for i in range(num_particles):
    assert(load_b.keys[i] >= 0 and load_b.keys[i] < total_keys)

# make sure that all particles are accounted in building the
# global tree
assert(load_b.global_num_real_particles == 125)

# the work is just the number of particles in each leaf
# so the sum of each leaf should be the total number of particles
assert(np.sum(load_b.global_work) == 125)

# the particle array should only have real particles
assert(pa.num_real_particles == pa.get_number_of_particles())

for i in xrange(pa.get_number_of_particles()):
    assert(abs(X[pa['gid'][i]] - pa['position-x'][i]) < 1e-15)
    assert(abs(Y[pa['gid'][i]] - pa['position-y'][i]) < 1e-15)
    assert(abs(Z[pa['gid'][i]] - pa['position-z'][i]) < 1e-15)
    assert(GID[pa['gid'][i]] == pa['gid'][i])
