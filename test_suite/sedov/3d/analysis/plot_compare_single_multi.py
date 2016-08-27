import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mat_colors
from matplotlib.collections import PatchCollection

fields = ['position-x', 'position-y', 'position-z',
    'velocity-x', 'velocity-y', 'velocity-z',
    'density', 'pressure',
    'volume', 'tag',
    'type', 'ids',
    'process']

file_name = '../multi_core/uniform/sedov_2d_uniform_output/' +\
        'sedov_2d_uniform_0113/data0113_cpu'
sedov_mc = phd.ParticleContainer(dim=3)

# stitch back multi-core solution
num_procs = 4
for i in range(num_procs):

    data_file = file_name + `i`.zfill(4) + '.hdf5'
    f = h5py.File(data_file, "r")
    size = f['/density'].size

    pc = phd.ParticleContainer(size, dim=3)
    for field in fields:
        pc[field][:] = f['/'+field][:]

    pc.remove_tagged_particles(phd.ParticleTAGS.Ghost)
    sedov_mc.append_container(pc)
    del pc

    f.close()

# single-core solution
file_name ='../single_core/uniform/sedov_3d_uniform_output/sedov_3d_uniform_0113.hdf5'
f = h5py.File(file_name, 'r')
sedov_sc = phd.ParticleContainer(f['/density'].size, dim=3)
for field in fields:
    sedov_sc[field][:] = f['/'+field][:]
f.close()
sedov_sc.remove_tagged_particles(phd.ParticleTAGS.Ghost)

# exact sedov solution
exact  = np.loadtxt("exact_sedov_3d.dat")
rad_ex = exact[:,1]
rho_ex = exact[:,2]
per_ex = exact[:,4]
vel_ex = exact[:,5]

fig, axes = plt.subplots(3,3, figsize=(12,12))

# put position and velocity in radial coordinates
rad_sc = np.sqrt((sedov_sc['position-x']-0.5)**2 + (sedov_sc['position-y']-0.5)**2 + (sedov_sc['position-z']-0.5)**2)
rad_mc = np.sqrt((sedov_mc['position-x']-0.5)**2 + (sedov_mc['position-y']-0.5)**2 + (sedov_mc['position-z']-0.5)**2)
vel_sc = np.sqrt(sedov_sc['velocity-x']**2 + sedov_sc['velocity-y']**2 + sedov_sc['velocity-z']**2)
vel_mc = np.sqrt(sedov_mc['velocity-x']**2 + sedov_mc['velocity-y']**2 + sedov_mc['velocity-z']**2)

# plot density curves
ax = axes[0,0]
ax.scatter(rad_mc, sedov_mc['density'], color='darkgray', label='multi-core')
ax.plot(rad_ex, rho_ex, "k")
ax.set_xlim(0,0.8)
ax.set_ylim(-1,7)
ax.set_xlabel('Radius')
ax.set_ylabel('Density')

ax = axes[0,1]
ax.scatter(rad_sc, sedov_sc['density'], color='steelblue', label='single-core')
ax.plot(rad_ex, rho_ex, "k")
ax.set_xlim(0,0.8)
ax.set_ylim(-1,7)
ax.set_xlabel('Radius')
ax.set_ylabel('Density')

ax = axes[2,0]
ax.scatter(rad_mc, sedov_mc['pressure'], color='darkgray', label='multi-core')
ax.plot(rad_ex, per_ex, "k")
ax.set_xlim(0,0.8)
ax.set_ylim(-0.5,2.5)
ax.set_xlabel('Radius')
ax.set_ylabel('Pressure')

ax = axes[1,0]
ax.scatter(rad_mc, vel_mc, color='darkgray', label='multi-core')
ax.plot(rad_ex, vel_ex, "k")
ax.set_xlim(0,0.8)
ax.set_ylim(-0.5,2.0)
ax.set_xlabel('Radius')
ax.set_ylabel('Velocity')

ax = axes[1,1]
ax.scatter(rad_sc, vel_sc, color='steelblue', label='single-core')
ax.plot(rad_ex, vel_ex, "k")
ax.set_xlim(0,0.8)
ax.set_ylim(-0.5,2.0)
ax.set_xlabel('Radius')
ax.set_ylabel('Velocity')

ax = axes[2,1]
ax.scatter(rad_sc, sedov_sc['pressure'], color='steelblue', label='single-core')
ax.plot(rad_ex, per_ex, "k")
ax.set_xlim(0,0.8)
ax.set_ylim(-0.5,2.5)
ax.set_xlabel('Radius')
ax.set_ylabel('Pressure')

ax = axes[0,2]
ids_sc = np.argsort(sedov_sc['ids'])
ids_mc = np.argsort(sedov_mc['ids'])
ax.scatter(sedov_sc['density'][ids_sc], sedov_mc['density'][ids_mc], marker='x', color='indianred')
ax.plot([0, 3], [0, 3], color='k')
ax.set_xlim(0,3)
ax.set_ylim(0,3)
ax.set_xlabel('Density (SC)')
ax.set_ylabel('Density (MC)')

ax = axes[1,2]
ax.scatter(vel_sc[ids_sc], vel_mc[ids_mc], marker='x', color='indianred')
ax.plot([-1, 2], [-1, 2], color='k')
ax.set_xlim(-0.1,1.4)
ax.set_ylim(-0.1,1.4)
ax.set_xlabel('Velocity (SC)')
ax.set_ylabel('Velocity (MC)')

ax = axes[2,2]
ax.scatter(sedov_sc['pressure'][ids_sc], sedov_mc['pressure'][ids_mc], marker='x', color='indianred')
ax.plot([-1, 3], [-1, 3], color='k')
ax.set_xlim(-0.1,1.8)
ax.set_ylim(-0.1,1.8)
ax.set_xlabel('Pressure (SC)')
ax.set_ylabel('Pressure (MC)')

plt.tight_layout()
plt.savefig("compare_single_multi_uniform.png")
plt.show()
