import phd
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16

reader = phd.Hdf5()
fig, axes = plt.subplots(2,2, figsize=(8,8))
file_names =["kelvin_output/time_interval/time_interval0004/time_interval0004.hdf5",
        "kelvin_output/time_interval/time_interval0009/time_interval0009.hdf5",
        "kelvin_output/time_interval/time_interval0014/time_interval0014.hdf5",
        "kelvin_output/final_output/final_output0000/final_output0000.hdf5"]

#for i, file_name in enumerate(file_names):
for i in range(2):
    for j in range(2):

        file_name = file_names[2*i + j]
        kelvin = reader.read(file_name)
        patch, colors = phd.vor_collection(kelvin, "density")
        kelvin.remove_tagged_particles(phd.ParticleTAGS.Ghost)

        time = h5py.File(file_name, "r").attrs["time"]

        p = PatchCollection(patch, edgecolor="gray", linewidth=0.1)
        p.set_array(np.array(colors))
        p.set_clim([0.9, 2.1])

        ax = axes[i, j]
        ax.text(0.03, 0.92, r"$t=%0.2f$" % time, fontsize=18,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.add_collection(p)
        ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("kelvin.eps")
plt.show()
