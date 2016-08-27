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

file_name = '../multi_core/cartesian/sod_2d_cartesian_output/' +\
        'sod_2d_cartesian_0072/data0072_cpu'
sod_mc = phd.ParticleContainer()

# stitch back multi-core solution
num_proc = 5
for i in range(num_proc):

    data_file = file_name + `i`.zfill(4) + '.hdf5'
    f = h5py.File(data_file, "r")
    size = f['/density'].size

    pc = phd.ParticleContainer(size)
    for field in fields:
        pc[field][:] = f['/'+field][:]

    sod_mc.append_container(pc)
    del pc

    f.close()

# single-core solution
file_name='../single_core/cartesian/sod_2d_cartesian_output/sod_2d_cartesian_0072.hdf5'
f = h5py.File(file_name, 'r')
sod_sc = phd.ParticleContainer(f['/density'].size)
for field in fields:
    sod_sc[field][:] = f['/'+field][:]
f.close()

# exact riemann solution
f = h5py.File("riemann_sol.hdf5", "r")
pos_ex = f['/x'][:]
rho_ex = f['/density'][:]
per_ex = f['/pressure'][:]
vel_ex = f['/velocity'][:]
f.close()

fig, axes = plt.subplots(3,3, figsize=(12,12))
patch, colors = phd.vor_collection(sod_sc, "density")
sod_sc.remove_tagged_particles(phd.ParticleTAGS.Ghost)

p = PatchCollection(patch, alpha=0.4)
p.set_array(np.array(colors))
p.set_clim([0, 1.0])
ax = axes[0,1]
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

# first remove interior particles
tag = np.where(sod_mc['type'] == phd.ParticleTAGS.Interior)[0]
sod_mc.remove_items(tag)

# create color map
cmap = mat_colors.ListedColormap(['green', 'red', 'blue', 'cyan', 'gray'])
bounds = [-1.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = mat_colors.BoundaryNorm(bounds, cmap.N)

patch, colors = phd.vor_collection(sod_mc, "process")
sod_mc.remove_tagged_particles(phd.ParticleTAGS.Ghost)
p = PatchCollection(patch, cmap=cmap, norm=norm, alpha=0.4)
p.set_array(np.array(colors))

ax = axes[0,0]
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

vel_sc = np.sqrt(sod_sc['velocity-x']**2 + sod_sc['velocity-y']**2)
vel_mc = np.sqrt(sod_mc['velocity-x']**2 + sod_mc['velocity-y']**2)

# plot density curves
ax = axes[1,0]
ax.scatter(sod_mc['position-x'], sod_mc['density'], color='darkgray', label='multi-core')
ax.plot(pos_ex, rho_ex, "k")
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.1)
ax.set_xlabel('X')
ax.set_ylabel('Density')

ax = axes[1,1]
ax.scatter(sod_sc['position-x'], sod_sc['density'], color='steelblue', label='single-core')
ax.plot(pos_ex, rho_ex, "k")
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.1)
ax.set_xlabel('X')
ax.set_ylabel('Density')

ax = axes[2,0]
ax.scatter(sod_mc['position-x'], vel_mc, color='darkgray', label='multi-core')
ax.plot(pos_ex, vel_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel('X')
ax.set_ylabel('Velocity')

ax = axes[2,1]
ax.scatter(sod_sc['position-x'], vel_sc, color='steelblue', label='single-core')
ax.plot(pos_ex, vel_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel('X')
ax.set_ylabel('Velocity')

ax = axes[1,2]
ids_sc = np.argsort(sod_sc['ids'])
ids_mc = np.argsort(sod_mc['ids'])
ax.scatter(sod_sc['density'][ids_sc], sod_mc['density'][ids_mc], marker='x', color='indianred')
ax.plot([0, 5], [0, 5], color='k')
ax.set_xlim(0,1.1)
ax.set_ylim(0,1.1)
ax.set_xlabel('Density (SC)')
ax.set_ylabel('Density (MC)')

ax = axes[2,2]
ax.scatter(vel_sc[ids_sc], vel_mc[ids_mc], marker='x', color='indianred')
ax.plot([-1, 2], [-1, 2], color='k')
ax.set_xlim(-0.1,1.1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel('Velocity (SC)')
ax.set_ylabel('Velocity (MC)')

ax = axes[0,2]
ax.scatter(np.log10(sod_sc['volume'][ids_sc]), np.log10(sod_mc['volume'][ids_mc]), marker='x', color='indianred')
ax.plot([-5, -1], [-5, -1], color='k')
ax.set_xlim(-3.8,-3.)
ax.set_ylim(-3.8,-3.)
ax.set_xlabel('log(Volume) (SC)')
ax.set_ylabel('log(Volume) (MC)')

plt.tight_layout()
plt.savefig("compare_single_multi_cartesian.pdf")
plt.show()
