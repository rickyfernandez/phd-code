import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt

file_names = []
file_ ='../multi_core/uniform/sedov_3d_uniform_output/sedov_3d_uniform_0113/data0113_cpu'

num_proc = 4
for i in range(num_proc):
    file_names.append(file_ + `i`.zfill(4) + '.hdf5')

# stitch back solution from all processors
particles = phd.ParticleContainer(dim=3)
for data_file in file_names:

    f = h5py.File(data_file, 'r')
    size = f['/density'].size
    pc = phd.ParticleContainer(size, dim=3)

    pc['position-x'][:] = f['/position-x'][:]
    pc['position-y'][:] = f['/position-y'][:]
    pc['position-z'][:] = f['/position-z'][:]
    pc['density'][:] = f['/density'][:]
    pc['velocity-x'][:] = f['/velocity-x'][:]
    pc['velocity-y'][:] = f['/velocity-y'][:]
    pc['velocity-z'][:] = f['/velocity-z'][:]
    pc['pressure'][:] = f['/pressure'][:]
    pc['tag'][:] = f['/tag'][:]
    pc['type'][:] = f['/type'][:]
    pc['volume'][:] = f['/volume'][:]

    pc.remove_tagged_particles(phd.ParticleTAGS.Ghost)
    particles.append_container(pc)

    f.close()

# get the exact solution
exact = np.loadtxt('exact_sedov_3d.dat')
x_ex = exact[:,1]
r_ex = exact[:,2]
p_ex = exact[:,4]
u_ex = exact[:,5]

r1 = np.sqrt((particles['position-x']-0.5)**2 + (particles['position-y']-0.5)**2 + (particles['position-z']-0.5)**2)
v = np.sqrt(particles['velocity-x']**2 + particles['velocity-y']**2 + particles['velocity-z']**2)

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.scatter(r1, particles['density'], color='lightsteelblue', label='phd')
plt.plot(x_ex, r_ex, 'k', label='exact')
plt.xlim(0,0.8)
plt.ylim(-1,7)
plt.xlabel('Position')
plt.ylabel('Density')
l = plt.legend(loc='upper left', prop={'size':12})
l.draw_frame(False)

plt.subplot(2,2,2)
plt.scatter(r1, v, color='lightsteelblue')
plt.plot(x_ex, u_ex, 'k')
plt.xlim(0,0.8)
plt.ylim(-0.5,2.0)
plt.xlabel('Position')
plt.ylabel('Velocity')

plt.subplot(2,2,3)
plt.scatter(r1, particles['pressure'], color='lightsteelblue')
plt.plot(x_ex, p_ex, 'k')
plt.xlim(0,0.8)
plt.ylim(-0.5,3.0)
plt.xlabel('Position')
plt.ylabel('Pressure')

plt.subplot(2,2,4)
plt.scatter(r1, particles['volume'], color='lightsteelblue')
plt.xlim(0,0.8)
plt.ylim(-0.02,0.03)
plt.xlabel('Position')
plt.ylabel('Volume')

plt.tight_layout()
plt.savefig('sedov_3d_multi_core.png')
plt.show()
