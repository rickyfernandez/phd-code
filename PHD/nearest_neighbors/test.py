import octree
import random
import hilbert
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# create particles 
num_particles = 128
particles = np.random.random(2*num_particles).reshape(2,num_particles).astype(np.float64)
#particles = np.random.normal(0.5, 0.1, 2*num_particles).reshape(2, num_particles).astype(np.float64)

# map particles to hilbert grid
order = 4
particles_h = np.array(particles * 2**order, dtype=np.int32)
keys = np.array([hilbert.hilbert_key_2d(p[0], p[1], order) for p in particles_h.T], dtype=np.int64)
sorted_indices = np.array(sorted(range(keys.shape[0]), key=lambda k: keys[k]))
sorted_particles = particles[:,sorted_indices]
sorted_keys = np.array(keys[sorted_indices], dtype=np.int64)


# create octree
tree = octree.Octree(sorted_keys, 4, order, 1, 1)
tree.build_tree()

# create plot
current_axis = plt.gca()
for node in tree.dump_data():
    x = node[0]/2.0**order
    y = node[1]/2.0**order
    w = node[2]/2.0**order
    current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, fill=None))

key_index = random.choice(range(sorted_keys.shape[0]))
key = sorted_keys[key_index]
node = tree.find_oct(key)
print node
x = node[0]/2.0**order
y = node[1]/2.0**order
w = node[2]/2.0**order
current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, color='red', alpha=0.5))
plt.plot(sorted_particles[0, key_index], sorted_particles[1, key_index], marker="*", ms=10, c='r')

#num_points = 2**order
#dx = 1.0/2**order
#points = [(x,y) for x in range(num_points) for y in range(num_points)]
#sorted_points = np.array(sorted(points, key=lambda k: hilbert.hilbert_key_2d(k[0], k[1], order)))
#plt.plot(sorted_points[:,0]/(2.0**order) + 0.5*dx, sorted_points[:,1]/(2.0**order) + 0.5*dx, 'r', alpha=0.5)

plt.scatter(particles[0,:], particles[1,:])
plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.savefig("oct.png")
plt.show()

