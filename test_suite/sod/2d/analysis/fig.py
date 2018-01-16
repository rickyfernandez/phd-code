import phd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

# single-core solution
file_name="../single_core/cartesian/sod_output/final_output/final_output0000/final_output0000.hdf5"
reader = phd.Hdf5()
sedov = reader.read(file_name)

fig, ax = plt.subplots(1,1, figsize=(8,8))
patch, colors = phd.vor_collection(sedov, "density")

p = PatchCollection(patch, alpha=0.4)
p.set_array(np.array(colors))
p.set_clim([0, 1.0])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.add_collection(p)

#plt.savefig("constant_hllc_static_cartesian_smooth.png")
plt.show()
