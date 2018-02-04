import h5py
import phd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mat_colors
from matplotlib.collections import PatchCollection

vol = 0.
num_proc = 4
reader = phd.Hdf5()

file_name="../single_core/random/sod_output/iteration_interval/iteration_interval0137/iteration_interval0137.hdf5"
sod_single = reader.read(file_name)

sod_multi = phd.CarrayContainer(0, sod_single.carray_dtypes)
for i in range(num_proc):

    file_name="../multi_core/random/sod_output/iteration_interval/iteration_interval0137_cpu" +\
            str(i).zfill(4) + "/iteration_interval0137_cpu" +\
            str(i).zfill(4) + ".hdf5"

    reader = phd.Hdf5()
    sod = reader.read(file_name)
    sod_multi.append_container(sod)

# exact riemann solution
f = h5py.File("riemann_sol.hdf5", "r")
pos_ex = f['/x'][:]
rho_ex = f['/density'][:]
per_ex = f['/pressure'][:]
vel_ex = f['/velocity'][:]
f.close()

fig, axes = plt.subplots(3,3, figsize=(12,12))
patch, colors = phd.vor_collection(sod_single, "velocity-x")
sod_single.remove_tagged_particles(phd.ParticleTAGS.Ghost)

p = PatchCollection(patch, alpha=0.4)
p.set_array(np.array(colors))
p.set_clim([0, 1.0])
ax = axes[0,1]
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

ax = axes[0,0]
for i in range(num_proc):

    file_name="../multi_core/random/sod_output/iteration_interval/iteration_interval0137_cpu" +\
            str(i).zfill(4) + "/iteration_interval0137_cpu" +\
            str(i).zfill(4) + ".hdf5"

    reader = phd.Hdf5()
    tmp = reader.read(file_name)
    patch, colors = phd.vor_collection(tmp, "velocity-x")

    p = PatchCollection(patch, alpha=0.4)
    p.set_array(np.array(colors))
    p.set_clim([0, 1.0])
    ax.add_collection(p)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,1)
ax.set_ylim(0,1)

sod_multi.remove_tagged_particles(phd.ParticleTAGS.Ghost)

ind1 = np.argsort(sod_single["ids"])
ind2 = np.argsort(sod_multi["ids"])
for field in sod_single.carrays.keys():
    sod_multi[field][:] = sod_multi[field][ind2]
    sod_single[field][:] = sod_single[field][ind1]
vel_sc = np.sqrt(sod_single['velocity-x']**2 + sod_single['velocity-y']**2)
vel_mc = np.sqrt(sod_multi['velocity-x']**2 + sod_multi['velocity-y']**2)
diff = []
for i in range(sod_single.get_carray_size()):
    if np.abs(vel_sc[i] - vel_mc[i]) > 1.0E-2:
        diff.append(i)
diff = np.array(diff)
if diff.size:
    ax.plot(sod_multi["position-x"][diff], sod_multi["position-y"][diff], "ro")

# plot density curves
ax = axes[1,0]
ax.scatter(sod_multi['position-x'], sod_multi['density'], color='darkgray', label='multi-core')
ax.plot(pos_ex, rho_ex, "k")
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.1)
ax.set_xlabel('X')
ax.set_ylabel('Density')

ax = axes[1,1]
ax.scatter(sod_single['position-x'], sod_single['density'], color='steelblue', label='single-core')
ax.plot(pos_ex, rho_ex, "k")
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.1)
ax.set_xlabel('X')
ax.set_ylabel('Density')

ax = axes[2,0]
ax.scatter(sod_multi['position-x'], vel_mc, color='darkgray', label='multi-core')
ax.plot(pos_ex, vel_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel('X')
ax.set_ylabel('Velocity')

ax = axes[2,1]
ax.scatter(sod_single['position-x'], vel_sc, color='steelblue', label='single-core')
ax.plot(pos_ex, vel_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel('X')
ax.set_ylabel('Velocity')

ax = axes[1,2]
ids_sc = np.argsort(sod_single['ids'])
ids_mc = np.argsort(sod_multi['ids'])
ax.scatter(sod_single['density'][ids_sc], sod_multi['density'][ids_mc], marker='x', color='indianred')
ax.plot([0, 5], [0, 5], color='k')
ax.set_xlim(0,1.1)
ax.set_ylim(0,1.1)
ax.set_xlabel('Density (SC)')
ax.set_ylabel('Density (MC)')

ax = axes[2,2]
#ax.scatter(vel_sc[ids_sc], vel_mc[ids_mc], marker='x', color='indianred')
ax.scatter(sod_single["velocity-x"], sod_multi["velocity-x"], marker='x', color='indianred')
if diff.size:
    ax.plot(sod_single["velocity-x"][diff], sod_multi["velocity-x"][diff], 'bo')
ax.plot([-1, 2], [-1, 2], color='k')
ax.set_xlim(-0.1,1.1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel('Velocity (SC)')
ax.set_ylabel('Velocity (MC)')

ax = axes[0,2]
ax.scatter(np.log10(sod_single['volume'][ids_sc]), np.log10(sod_multi['volume'][ids_mc]), marker='x', color='indianred')
ax.plot([-4.6, -3.6], [-4.6, -3.6], color='k')
ax.set_xlim(-4.6,-3.6)
ax.set_ylim(-4.6,-3.6)
ax.set_xlabel('log(Volume) (SC)')
ax.set_ylabel('log(Volume) (MC)')
plt.tight_layout()
#plt.savefig("compare_single_multi_random.pdf")
plt.show()
