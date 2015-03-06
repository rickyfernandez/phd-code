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
import h5py

from load_balance.load_balance import LoadBalance

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if size != 4:
    if rank == 0:
        raise RuntimeError("Run this test with 4 processors")

# create the initial distribution
if rank == 0:
    numPoints = 12
    x = np.array( [0.0, 1.0, 2.0, 3.0, 4.0,
                   0.0, 1.0, 2.0, 3.0, 4.0,
                   0.0, 1.0] )

    y = np.array( [0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, 1.0, 1.0, 1.0, 1.0,
                   2.0, 2.0] )

    points = np.array([x,y], dtype=np.float64)
    gid = np.array( [0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 10,
                     11], dtype=np.int32 )

if rank == 1:
    numPoints = 6
    x = np.array( [2.0, 3.0, 4.0,
                   0.0, 1.0, 2.0] )

    y = np.array( [2.0, 2.0, 2.0,
                   3.0, 3.0, 3.0] )

    points = np.array([x,y], dtype=np.float64)
    gid = np.array( [12, 13, 14, 15,
                     16, 17], dtype=np.int32 )

if rank == 2:
    numPoints = 3
    x = np.array( [4.0, 3.0, 0.0] )
    y = np.array( [3.0, 3.0, 4.0] )

    points = np.array([x,y], dtype=np.float64)
    gid = np.array( [18, 19, 20], dtype=np.int32 )


if rank == 3:
    numPoints = 4
    x = np.array( [1.0, 2.0, 3.0, 4.0] )
    y = np.array( [4.0, 4.0, 4.0, 4.0] )

    points = np.array([x,y], dtype=np.float64)
    gid = np.array( [21, 22, 23, 24], dtype=np.int32 )


f = h5py.File("initial_particle_data_" + `rank`.zfill(4) + ".hdf5", "w")
f['/x'] = x
f['/y'] = y
f.attrs["rank"] = rank
f.close

load_b = LoadBalance(points, order=3)
exchange_data = load_b.decomposition()

## Gather the Global data on root
X = np.zeros(shape=25, dtype=np.float64)
Y = np.zeros(shape=25, dtype=np.float64)
GID = np.zeros(shape=25, dtype=np.int32)

displacements = np.array( [12, 6, 3, 4], dtype=np.int32 )

comm.Gatherv(sendbuf=[x, MPI.DOUBLE], recvbuf=[X, (displacements, None)], root=0)
comm.Gatherv(sendbuf=[y, MPI.DOUBLE], recvbuf=[Y, (displacements, None)], root=0)
comm.Gatherv(sendbuf=[gid, MPI.INT], recvbuf=[GID, (displacements, None)], root=0)

if rank == 0:
    f = h5py.File("initial_global_particle_data.hdf5", "w")
    f['/x'] = X
    f['/y'] = Y
    f['/id'] = GID
    f.close

f = h5py.File("final_particle_data_" + `rank`.zfill(4) + ".hdf5", "w")
f['/global_id_to_send'] = gid[exchange_data["ids"]]
tmp = np.delete(gid, exchange_data["ids"])
if tmp.size == 0:
    f['/global_id_to_stay'] = np.array([-1, -1])
else:
    f['/global_id_to_stay'] = tmp
f['/to_proc'] = exchange_data["procs"]
f.attrs["rank"] = rank
f.close
