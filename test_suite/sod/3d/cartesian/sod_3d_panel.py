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
file_name="sod_output/final_output/final_output0000/final_output0000.hdf5"
reader = phd.Hdf5()
sod = reader.read(file_name)

# exact riemann solution
f = h5py.File("riemann_sol.hdf5", "r")
pos_ex = f["/x"][:]
rho_ex = f["/density"][:]
pre_ex = f["/pressure"][:]
vel_ex = f["/velocity"][:]
f.close()

fig, axes = plt.subplots(3, 1, sharex="col", figsize=(6,12))
fig.subplots_adjust(hspace=0, left=0.13, top=0.96, bottom=0.06)

# we plot a slice at z=0
end=45**2

ax = axes[0]
ax.plot(sod["position-x"][0:end], sod["density"][0:end], ".", color="steelblue")
ax.plot(pos_ex, rho_ex, "red", label="exact")
ax.set_xlim(0,1)
ax.set_ylim(0,1.2)
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_yticklabels([0.0, 0.5, 1.0])
ax.set_ylabel(r"$\rho$", fontsize=18)
ax.tick_params(direction="in", right=True, top=True)
ax.set_title("3D", fontsize=18)

ax = axes[1]
ax.plot(sod["position-x"][0:end], sod["velocity-x"][0:end], ".", color="steelblue")
ax.plot(pos_ex, vel_ex, "red")
ax.set_xlim(0,1)
ax.set_ylim(-0.2,1.2)
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_yticklabels([0.0, 0.5, 1.0])
ax.set_ylabel(r"$v_x$", fontsize=18)
ax.tick_params(direction="in", right=True, top=True)

ax = axes[2]
ax.plot(sod["position-x"][0:end], sod["pressure"][0:end], ".", color="steelblue")
ax.plot(pos_ex, pre_ex, "red")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.2)
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_yticklabels([0.0, 0.5, 1.0])
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$P$", fontsize=18)
ax.tick_params(direction="in", right=True, top=True)

plt.savefig("sod_3d.eps")
plt.show()
