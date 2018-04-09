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

# single-core solution
static_names=[
        ["rayleigh_static_output/time_interval/time_interval0000/time_interval0000.hdf5",
         "rayleigh_static_output/time_interval/time_interval0001/time_interval0001.hdf5",
         "rayleigh_static_output/time_interval/time_interval0002/time_interval0002.hdf5",
         "rayleigh_static_output/time_interval/time_interval0003/time_interval0003.hdf5"],
        ["rayleigh_moving_output/time_interval/time_interval0000/time_interval0000.hdf5",
         "rayleigh_moving_output/time_interval/time_interval0001/time_interval0001.hdf5",
         "rayleigh_moving_output/time_interval/time_interval0002/time_interval0002.hdf5",
         "rayleigh_moving_output/time_interval/time_interval0003/time_interval0003.hdf5"]]

fig, axes = plt.subplots(2,4, figsize=(8,12), sharey=True, sharex=True)
plt.subplots_adjust(wspace=0.1, hspace=0.04, top=0.98, bottom=0.02, left=0.02, right=0.98)

reader = phd.Hdf5()
for i in range(4): # over columns
    for j in range(2): # over plots

        file_name = static_names[j][i]
        particles = reader.read(file_name)
        time = h5py.File(file_name, "r").attrs["time"]

        ax = axes[j,i]
        patch, colors = phd.vor_collection(particles, "density")
        particles.remove_tagged_particles(phd.ParticleTAGS.Ghost)

        print particles["density"].min(), particles["density"].max()

        p = PatchCollection(patch, cmap="jet", edgecolor="none")
        p.set_array(np.array(colors))
        p.set_clim([0.9, 2.1])
        ax.add_collection(p)
        ax.set_xlim(0,1)
        ax.set_ylim(0,3)

        ax.text(0.07, 2.80, r"$t=%0.2f$" % time, fontsize=14,
                bbox=dict(boxstyle="round", facecolor="white"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("auto")

#plt.savefig("rayleigh_compare.eps")
plt.show()
