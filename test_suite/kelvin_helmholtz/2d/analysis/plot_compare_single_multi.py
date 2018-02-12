import h5py
import phd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mat_colors
from matplotlib.collections import PatchCollection

num_proc = 4
reader = phd.Hdf5()

file_name="../single_core/kh_output/final_output/final_output0000/final_output0000.hdf5"
kh_single = reader.read(file_name)

fig, axes = plt.subplots(2,2, figsize=(12,12))
patch, colors = phd.vor_collection(kh_single, "density")
kh_single.remove_tagged_particles(phd.ParticleTAGS.Ghost)

p = PatchCollection(patch, edgecolors="none",cmap="jet")
p.set_array(np.array(colors))
p.set_clim([0.7, 2.1])
ax = axes[0,1]
ax.set_title("Single Core")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

ax = axes[0,0]
ax.set_title("Multi Core")
kh_multi = phd.CarrayContainer(0, kh_single.carray_dtypes)
for i in range(num_proc):

    file_name="../multi_core/kh_output/final_output/final_output0000_cpu" +\
            str(i).zfill(4) + "/final_output0000_cpu" +\
            str(i).zfill(4) + ".hdf5"

    reader = phd.Hdf5()
    kh = reader.read(file_name)

    patch, colors = phd.vor_collection(kh, "density")
    p = PatchCollection(patch, edgecolors="none",cmap="jet")
    p.set_array(np.array(colors))
    p.set_clim([0.7, 2.1])
    ax.add_collection(p)

    kh_multi.append_container(kh)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,1)
ax.set_ylim(0,1)

# sort data by particle ids
kh_multi.remove_tagged_particles(phd.ParticleTAGS.Ghost)
kh_single.remove_tagged_particles(phd.ParticleTAGS.Ghost)
ind1 = np.argsort(kh_single["ids"])
ind2 = np.argsort(kh_multi["ids"])
for field in kh_single.carrays.keys():
    kh_multi[field][:] = kh_multi[field][ind2]
    kh_single[field][:] = kh_single[field][ind1]

# plot density curves
ax = axes[1,0]
ax.set_title("Density")
ax.scatter(kh_single["density"], kh_multi["density"], color="darkgray", label="multi-core")
ax.set_xlabel("Density (SC)")
ax.set_ylabel("Density (MC)")

ax = axes[1,1]
ax.set_title("Pressure")
ax.scatter(kh_single["pressure"], kh_multi["pressure"], color="darkgray", label="multi-core")
ax.set_xlabel("Pressure (SC)")
ax.set_ylabel("Pressure (MC)")

plt.savefig("compare_kh.pdf")
plt.show()
