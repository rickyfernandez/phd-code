from mpi4py import MPI
import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from particles.particle_container import ParticleContainer
from load_balance.load_balance import LoadBalance

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# generate random particles in a unit box for each process
np.random.seed(rank)
#my_num_particles = np.random.randint(64, 256)
my_num_particles = np.random.randint(256, 512)
my_particles = np.random.random(2*my_num_particles).reshape(2, my_num_particles).astype(np.float64)
#my_particles = np.random.normal(0.5, 0.1, 2*my_num_particles).reshape(2, my_num_particles).astype(np.float64)

pc = ParticleContainer(my_num_particles)
pc['position-x'][:] = my_particles[0,:]
pc['position-y'][:] = my_particles[1,:]

#plot initial distribuition
plt.scatter(pc['position-x'], pc['position-y'])
plt.savefig("plot_init_proc_%d.png" % rank)
plt.clf()

# perform load balance
order = 21
lb = LoadBalance(pc, comm=comm)
lb.decomposition()
lb.exchange_particles()
num_particles = pc.num_real_particles

#print 'rank: %d number of particles %d: number of leaves %d' % (rank, pc.num_real_particles, lb.global_work.size)

#lb.create_ghost_particles()
#current_axis = plt.gca()
# plot bounding leaf
#for node in p:
#    x = node[0]/2.0**order
#    y = node[1]/2.0**order
##    w = node[2]/2.0**order
#    plt.plot(x, y, 'r*')
##    current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, color='orange', alpha=0.5))
#
# plot global tree
current_axis = plt.gca()
for node in lb.global_tree.dump_data():
    x = node[0]/2.0**order
    y = node[1]/2.0**order
    w = node[2]/2.0**order
    current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, fill=None))

# create border particles
border = lb.create_ghost_particles()

# plot particles
plt.scatter(pc['position-x'][:num_particles], pc['position-y'][:num_particles])
plt.scatter(pc['position-x'][num_particles:], pc['position-y'][num_particles:], color="red")
plt.scatter(pc['position-x'][border], pc['position-y'][border], color="green")
#plt.scatter(send_data['position-x'], send_data['position-y'], color="red")
plt.xlim(-0.2,1.2)
plt.ylim(-0.2,1.2)
plt.savefig("plot_proc_%d.png" % rank)
