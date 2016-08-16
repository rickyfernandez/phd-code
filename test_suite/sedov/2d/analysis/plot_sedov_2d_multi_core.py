import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt

file_names = []
file_ = '../multi_core/cartesian/sedov_2d_cartesian_output/sedov_2d_cartesian_0139/data0139_cpu'
for i in range(4):
    file_names.append(file_ + `i`.zfill(4) + '.hdf5')

# stitch back solution from all processors
particles = phd.ParticleContainer()
for data_file in file_names:

    f = h5py.File(data_file, 'r')
    size = f['/density'].size
    pc = phd.ParticleContainer(size)

    pc['position-x'][:] = f['/position-x'][:]
    pc['position-y'][:] = f['/position-y'][:]
    pc['density'][:] = f['/density'][:]
    pc['velocity-x'][:] = f['/velocity-x'][:]
    pc['velocity-y'][:] = f['/velocity-y'][:]
    pc['pressure'][:] = f['/pressure'][:]
    pc['tag'][:] = f['/tag'][:]
    pc['type'][:] = f['/type'][:]
    pc['volume'][:] = f['/volume'][:]

    pc.remove_tagged_particles(phd.ParticleTAGS.Ghost)
    particles.append_container(pc)

    f.close()

# get the exact solution
exact = np.loadtxt('exact_sedov_2d.dat')
x_ex = exact[:,1]
r_ex = exact[:,2]
p_ex = exact[:,4]
u_ex = exact[:,5]

r1 = np.sqrt((particles['position-x']-0.5)**2 + (particles['position-y']-0.5)**2)

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.scatter(r1, particles['density'], color='lightsteelblue', label='phd')
plt.plot(x_ex, r_ex, 'k', label='exact')
plt.xlim(0,0.5)
plt.ylim(-1,7)
plt.xlabel('Position')
plt.ylabel('Density')
l = plt.legend(loc='upper left', prop={'size':12})
l.draw_frame(False)

plt.subplot(2,2,2)
plt.scatter(r1, np.sqrt(particles['velocity-x']**2 + particles['velocity-y']**2), color='lightsteelblue')
plt.plot(x_ex, u_ex, 'k')
plt.xlim(0,0.5)
plt.ylim(-0.5,2.0)
plt.xlabel('Position')
plt.ylabel('Velocity')

plt.subplot(2,2,3)
plt.scatter(r1, particles['pressure'], color='lightsteelblue')
plt.plot(x_ex, p_ex, 'k')
plt.xlim(0,0.5)
plt.ylim(-0.5,3.0)
plt.xlabel('Position')
plt.ylabel('Pressure')

plt.subplot(2,2,4)
plt.scatter(r1, particles['volume'], color='lightsteelblue')
plt.xlim(0,0.5)
plt.ylim(-0.02,0.03)
plt.xlabel('Position')
plt.ylabel('Volume')

plt.tight_layout()
plt.savefig('sedov_2d_multi_core.pdf')
plt.show()
