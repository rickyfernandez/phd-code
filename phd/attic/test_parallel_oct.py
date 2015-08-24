import octree
import random
import hilbert
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from matplotlib.patches import Rectangle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

order = 4

if rank == 0:

    # create particles 
    num_particles = 11
    np.random.seed(0)
    particles = np.random.random(2*num_particles).reshape(2,num_particles).astype(np.float64)

    # map paricles to hilbert grid
    particles_h = np.array(particles * 2**order, dtype=np.int32)

    # generate hilber keys for each particles
    keys = np.array([hilbert.hilbert_key_2d(p[0], p[1], order) for p in particles_h.T], dtype=np.int32)

    # sort indices
    sorted_indices = np.array(sorted(range(keys.shape[0]), key=lambda k: keys[k]))
    sorted_particles = np.array(particles[:,sorted_indices])
    sorted_keys = np.array(keys[sorted_indices], dtype=np.int32)

    print "process %d sorted keys %s" % (rank, sorted_keys)

    # split keys among processes
    num_keys, remainder = divmod(sorted_keys.size, size)

    scounts = size*[num_keys]
    scounts[-1] += remainder
    counts = list(scounts)

    print "process %d scounts %s" % (rank, scounts)

    sdispls = size*[0]
    for i in xrange(1,size):
        sdispls[i] = scounts[i-1] + sdispls[i-1]

    print "process %d sdispls %s" % (rank, sdispls)

else:
    sorted_keys = None
    scounts = None
    sdispls = None
    counts = None


# tell the processors how much data is coming
counts = comm.scatter(counts, root=0)
keys_sorted_local = np.zeros(counts, dtype=np.int32)
print "process %d counts %s" % (rank, counts)

# send keys
comm.Scatterv([sorted_keys, scounts, sdispls, MPI.INT], [keys_sorted_local, MPI.INT])
print "processor %d has keys: %s" % (rank, keys_sorted_local)



# create octree
tree = octree.Octree(sorted_keys, 4, order)
tree.partial_build()
#
## create plot
#current_axis = plt.gca()
#for node in tree.dump_data():
#    x = node[0]/2.0**order
#    y = node[1]/2.0**order
#    w = node[2]/2.0**order
#    current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, fill=None))
#
##key_index = random.choice(range(sorted_keys.shape[0]))
#key_index = 68
#key = sorted_keys[key_index]
#node = tree.find_oct(key)
#
#x = node[0]/2.0**order
#y = node[1]/2.0**order
#w = node[2]/2.0**order
#current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, color='red', alpha=0.5))
#plt.plot(sorted_particles[0, key_index], sorted_particles[1, key_index], marker="*", ms=10, c='r')
#
## add neighbor nodes
#neighbors = tree.oct_neighbor_search(key)
#for node in neighbors:
#    x = node[0]/2.0**order
#    y = node[1]/2.0**order
#    w = node[2]/2.0**order
#    current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, color='orange', alpha=0.5))
#
#plt.scatter(particles[0,:], particles[1,:])
#plt.xlim(-0.05,1.05)
#plt.ylim(-0.05,1.05)
#plt.savefig("oct.png")
#plt.show()
#
