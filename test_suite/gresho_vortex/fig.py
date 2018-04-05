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

# create exact solution
radius = np.linspace(0, 0.5, 100)
sol = np.zeros(radius.size)

for i in range(sol.size):
    r = radius[i]
    if 0 <= r < 0.2:
        sol[i] = 5*r
    elif 0.2 <= r < 0.4:
        sol[i] = 2 - 5*r
    else:
        sol[i] = 0.
        
files = [
        "gresho_output/initial_output/initial_output0000/initial_output0000.hdf5",
        "gresho_output/time_interval/time_interval0000/time_interval0000.hdf5",
        "gresho_output/final_output/final_output0000/final_output0000.hdf5"
        ]

fig, axes = plt.subplots(2,3, figsize=(14,8))

for i, fi in enumerate(files):

    reader = phd.Hdf5()
    vortex = reader.read(fi)
    time = h5py.File(fi, "r").attrs["time"]

    vortex.register_carray(vortex.get_carray_size(),
            "v-theta", "double")
    vortex.register_carray(vortex.get_carray_size(),
            "rad", "double")

    # put position and velocity in radial coordinates
    theta = np.arctan2(vortex["position-y"]-0.5, vortex["position-x"]-0.5)
    vortex["rad"][:] = np.sqrt((vortex["position-x"]-0.5)**2 + (vortex["position-y"]-0.5)**2)
    vortex["v-theta"][:] = (-np.sin(theta)*vortex["velocity-x"] + np.cos(theta)*vortex["velocity-y"])

    # third column
    patch, colors = phd.vor_collection(vortex, "v-theta")
    vortex.remove_tagged_particles(phd.ParticleTAGS.Ghost)

    p = PatchCollection(patch, edgecolor="none", linewidth=0.1, cmap="jet", alpha=0.9)
    p.set_array(np.array(colors))
    p.set_clim([0, 1.0])
    ax = axes[0,i]
    ax.text(0.03, 0.92, r"$t=%0.2f$" % time, fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white"))
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.add_collection(p)

    ax = axes[1,i]
    ax.plot(vortex["rad"], vortex["v-theta"], ".", color="steelblue", label="simulation")
    ax.plot(radius, sol, "red")
    ax.set_ylabel(r"$v_\theta$", fontsize=18)
    ax.set_xlabel(r"$r$", fontsize=18)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(0,0.5)
    ax.set_ylim(0,1)
    ax.tick_params(direction="in", right=True, top=True)

fig.tight_layout()
plt.savefig("gresho_vortex.eps")
plt.show()
