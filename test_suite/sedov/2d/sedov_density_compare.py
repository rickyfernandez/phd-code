import phd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16

# single-core solution
file_name_2d="sedov_output/final_output/final_output0000/final_output0000.hdf5"
file_name_3d="../3d/single_core/cartesian/sedov_output/final_output/final_output0000/final_output0000.hdf5"
reader = phd.Hdf5()
sedov_2d = reader.read(file_name_2d)
sedov_3d = reader.read(file_name_3d)

# exact sedov solution
exact  = np.loadtxt("sedov_2d.dat")
rad_ex = exact[:,1]
rho_ex = exact[:,2]
pre_ex = exact[:,4]
vel_ex = exact[:,5]

fig, axes = plt.subplots(1,2, figsize=(12,6))

rad_2d = np.sqrt((sedov_2d["position-x"]-0.5)**2 + (sedov_2d["position-y"]-0.5)**2)
rad_3d = np.sqrt((sedov_3d["position-x"]-0.5)**2 + (sedov_3d["position-y"]-0.5)**2 + (sedov_3d["position-z"]-0.5)**2)

ax = axes[0]
ax.set_title("2D", fontsize=18)
ax.plot(rad_2d, sedov_2d["density"], ".", color="steelblue")
ax.plot(rad_ex, rho_ex, "r", label="exact")
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels([0, 1, 2, 3, 4])
ax.set_xlim(0,0.5)
ax.set_ylim(-0.1,4.1)
ax.set_xlabel(r"$r$", fontsize=18)
ax.set_ylabel(r"$\rho$", fontsize=18)
ax.tick_params(direction="in", right=True, top=True)

# exact sedov solution
exact  = np.loadtxt("../3d/single_core/cartesian/sedov_3d.dat")
rad_ex = exact[:,1]
rho_ex = exact[:,2]
pre_ex = exact[:,4]
vel_ex = exact[:,5]

np.random.seed(0)
ids = np.random.choice(sedov_3d.get_carray_size(), sedov_2d.get_carray_size())

ax = axes[1]
ax.set_title("3D", fontsize=18)
ax.plot(rad_3d[ids], sedov_3d["density"][ids], ".", color="steelblue")
ax.plot(rad_ex, rho_ex, "r", label="exact")
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels([0, 1, 2, 3, 4])
ax.set_xlim(0,0.5)
ax.set_ylim(-0.1,4.1)
ax.set_xlabel(r"$r$", fontsize=18)
ax.set_ylabel(r"$\rho$", fontsize=18)
ax.tick_params(direction="in", right=True, top=True)

plt.tight_layout()
plt.savefig("sedov_compare.eps")
plt.show()
