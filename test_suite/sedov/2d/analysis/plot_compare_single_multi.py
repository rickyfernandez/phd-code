import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mat_colors
from matplotlib.collections import PatchCollection

fields = ['position-x', 'position-y',
    'velocity-x', 'velocity-y',
    'density', 'pressure',
    'volume', 'tag',
    'type', 'ids',
    'process']

#file_name = '../multi_core/cartesian/sedov_2d_cartesian_output/' +\
#        'sedov_2d_cartesian_0139/data0139_cpu'
file_name = '../multi_core/uniform/sedov_2d_uniform_output/' +\
        'sedov_2d_uniform_0105/data0105_cpu'
sedov_mc = phd.ParticleContainer()

# stitch back multi-core solution
for i in range(5):

    data_file = file_name + `i`.zfill(4) + '.hdf5'
    f = h5py.File(data_file, "r")
    size = f['/density'].size

    pc = phd.ParticleContainer(size)
    for field in fields:
        pc[field][:] = f['/'+field][:]

    sedov_mc.append_container(pc)
    del pc

    f.close()

# single-core solution
#file_name ='../single_core/cartesian/sedov_2d_cartesian_output/sedov_2d_cartesian_0139.hdf5'
file_name ='../single_core/uniform/sedov_2d_uniform_output/sedov_2d_uniform_0105.hdf5'
f = h5py.File(file_name, 'r')
sedov_sc = phd.ParticleContainer(f['/density'].size)
for field in fields:
    sedov_sc[field][:] = f['/'+field][:]
f.close()

# exact sedov solution
exact  = np.loadtxt("exact_sedov_2d.dat")
rad_ex = exact[:,1]
rho_ex = exact[:,2]
per_ex = exact[:,4]
vel_ex = exact[:,5]

fig, axes = plt.subplots(3,3, figsize=(12,12))
patch, colors = phd.vor_collection(sedov_sc, "density")
sedov_sc.remove_tagged_particles(phd.ParticleTAGS.Ghost)

p = PatchCollection(patch, alpha=0.4)
p.set_array(np.array(colors))
p.set_clim([0, 4.0])
ax = axes[0,1]
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

# first remove interior particles
tag = np.where(sedov_mc['type'] == phd.ParticleTAGS.Interior)[0]
sedov_mc.remove_items(tag)

# create color map
cmap = mat_colors.ListedColormap(['green', 'red', 'blue', 'cyan', 'gray'])
bounds = [-1.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = mat_colors.BoundaryNorm(bounds, cmap.N)

patch, colors = phd.vor_collection(sedov_mc, "process")
sedov_mc.remove_tagged_particles(phd.ParticleTAGS.Ghost)
p = PatchCollection(patch, cmap=cmap, norm=norm, alpha=0.4)
p.set_array(np.array(colors))

ax = axes[0,0]
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

# put position and velocity in radial coordinates
rad_sc = np.sqrt((sedov_sc['position-x']-0.5)**2 + (sedov_sc['position-y']-0.5)**2)
rad_mc = np.sqrt((sedov_mc['position-x']-0.5)**2 + (sedov_mc['position-y']-0.5)**2)
vel_sc = np.sqrt(sedov_sc['velocity-x']**2 + sedov_sc['velocity-y']**2)
vel_mc = np.sqrt(sedov_mc['velocity-x']**2 + sedov_mc['velocity-y']**2)

# plot density curves
ax = axes[1,0]
ax.scatter(rad_mc, sedov_mc['density'], color='darkgray', label='multi-core')
ax.plot(rad_ex, rho_ex, "k")
ax.set_xlim(0,0.5)
ax.set_ylim(-1,7)
ax.set_xlabel('Radius')
ax.set_ylabel('Density')

ax = axes[1,1]
ax.scatter(rad_sc, sedov_sc['density'], color='steelblue', label='single-core')
ax.plot(rad_ex, rho_ex, "k")
ax.set_xlim(0,0.5)
ax.set_ylim(-1,7)
ax.set_xlabel('Radius')
ax.set_ylabel('Density')

ax = axes[2,0]
ax.scatter(rad_mc, vel_mc, color='darkgray', label='multi-core')
ax.plot(rad_ex, vel_ex, "k")
ax.set_xlim(0,0.5)
ax.set_ylim(-0.5,2)
ax.set_xlabel('Radius')
ax.set_ylabel('Velocity')

ax = axes[2,1]
ax.scatter(rad_sc, vel_sc, color='steelblue', label='single-core')
ax.plot(rad_ex, vel_ex, "k")
ax.set_xlim(0,0.5)
ax.set_ylim(-0.5,2)
ax.set_xlabel('Radius')
ax.set_ylabel('Velocity')

ax = axes[1,2]
ids_sc = np.argsort(sedov_sc['ids'])
ids_mc = np.argsort(sedov_mc['ids'])
ax.scatter(sedov_sc['density'][ids_sc], sedov_mc['density'][ids_mc], marker='x', color='indianred')
ax.plot([0, 5], [0, 5], color='k')
ax.set_xlim(0,4)
ax.set_ylim(0,4)
ax.set_xlabel('Density (SC)')
ax.set_ylabel('Density (MC)')

ax = axes[2,2]
ax.scatter(vel_sc[ids_sc], vel_mc[ids_mc], marker='x', color='indianred')
ax.plot([0, 2], [0, 2], color='k')
ax.set_xlim(0,1.5)
ax.set_ylim(0,1.5)
ax.set_xlabel('Velocity (SC)')
ax.set_ylabel('Velocity (MC)')

ax = axes[0,2]
ax.scatter(np.log10(sedov_sc['volume'][ids_sc]), np.log10(sedov_mc['volume'][ids_mc]), marker='x', color='indianred')
ax.plot([-5, -1], [-5, -1], color='k')
ax.set_xlim(-5,-1)
ax.set_ylim(-5,-1)
ax.set_xlabel('log(Volume) (SC)')
ax.set_ylabel('log(Volume) (MC)')

plt.tight_layout()
plt.savefig("compare_single_multi_uniform.pdf")
plt.show()
