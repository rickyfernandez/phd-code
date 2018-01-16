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
    'type', 'ids']
    #'process']

file_name='../single_core/cartesian/sod_2d_cartesian_output/sod_2d_cartesian_0164.hdf5'
#file_name='../single_core/uniform/sod_2d_uniform_output/sod_2d_uniform_0099.hdf5'
f = h5py.File(file_name, 'r')
#sod_sc = phd.ParticleContainer(f['/density'].size)
sod_sc = phd.HydroParticleCreator(f['/density'].size)
for field in fields:
    sod_sc[field][:] = f['/'+field][:]
f.close()

# exact riemann solution
f = h5py.File("riemann_sol.hdf5", "r")
pos_ex = f['/x'][:]
rho_ex = f['/density'][:]
pre_ex = f['/pressure'][:]
vel_ex = f['/velocity'][:]
f.close()

fig, axes = plt.subplots(2,2, figsize=(12,12))
plt.suptitle('HLL : Linear Reconstruction : Static')

patch, colors = phd.vor_collection(sod_sc, "density")
sod_sc.remove_tagged_particles(phd.ParticleTAGS.Ghost)

p = PatchCollection(patch, alpha=0.4)
p.set_array(np.array(colors))
p.set_clim([0, 1.0])
ax = axes[0,0]
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

vel_sc = np.sqrt(sod_sc['velocity-x']**2 + sod_sc['velocity-y']**2)

ax = axes[0,1]
ax.scatter(sod_sc['position-x'], sod_sc['density'], color='steelblue', label='single-core')
ax.plot(pos_ex, rho_ex, "k")
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.1)
ax.set_xlabel('X')
ax.set_ylabel('Density')

ax = axes[1,0]
ax.scatter(sod_sc['position-x'], vel_sc, color='steelblue', label='single-core')
ax.plot(pos_ex, vel_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel('X')
ax.set_ylabel('Velocity')

ax = axes[1,1]
ax.scatter(sod_sc['position-x'], sod_sc['pressure'], color='steelblue', label='single-core')
ax.plot(pos_ex, pre_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel('X')
ax.set_ylabel('Pressure')

#plt.tight_layout()
#plt.savefig("sod_linear_static_uniform_releax_10.png")
plt.savefig("test.pdf")
plt.show()
