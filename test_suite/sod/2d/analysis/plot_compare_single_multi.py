import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mat_colors
from matplotlib.collections import PatchCollection

io = phd.Hdf5()

# single-core solution
base_name = "random/sod_output/final_output/final_output0000"
file_name = "../single_core/" + base_name + "/final_output0000.hdf5"
sod_sc = io.read(file_name)

# stitch back multi-core solution
num_proc = 4
for i in range(num_proc):

    cpu = "_cpu" + str(i).zfill(4)
    file_name = "../multi_core/" + base_name + cpu + "/final_output0000" + cpu + ".hdf5"

    if i == 0:
        sod_mc = io.read(file_name)
    else:
        sod_mc.append_container(io.read(file_name))

# exact riemann solution
f = h5py.File("riemann_sol.hdf5", "r")
pos_ex = f["/x"][:]
rho_ex = f["/density"][:]
per_ex = f["/pressure"][:]
vel_ex = f["/velocity"][:]
f.close()

fig, axes = plt.subplots(3,3, figsize=(12,12))
patch, colors = phd.vor_collection(sod_sc, "density")
sod_sc.remove_tagged_particles(phd.ParticleTAGS.Ghost)

p = PatchCollection(patch, alpha=0.4)
p.set_array(np.array(colors))
p.set_clim([0, 1.0])
ax = axes[0,1]
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

# first remove interior particles
tag = np.where(sod_mc["type"] == phd.ParticleTAGS.Interior)[0]
sod_mc.remove_items(tag)

patch, colors = phd.vor_collection(sod_mc, "density")
sod_mc.remove_tagged_particles(phd.ParticleTAGS.Ghost)
p = PatchCollection(patch, alpha=0.4)
p.set_array(np.array(colors))
p.set_clim([0, 1.0])
ax = axes[0,0]
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

vel_sc = np.sqrt(sod_sc["velocity-x"]**2 + sod_sc["velocity-y"]**2)
vel_mc = np.sqrt(sod_mc["velocity-x"]**2 + sod_mc["velocity-y"]**2)

# plot density curves
ax = axes[1,0]
ax.scatter(sod_mc["position-x"], sod_mc["density"], color="darkgray", label="multi-core")
ax.plot(pos_ex, rho_ex, "k")
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.1)
ax.set_xlabel("X")
ax.set_ylabel("Density")

ax = axes[1,1]
ax.scatter(sod_sc["position-x"], sod_sc["density"], color="steelblue", label="single-core")
ax.plot(pos_ex, rho_ex, "k")
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.1)
ax.set_xlabel("X")
ax.set_ylabel("Density")

ax = axes[2,0]
ax.scatter(sod_mc["position-x"], vel_mc, color="darkgray", label="multi-core")
ax.plot(pos_ex, vel_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel("X")
ax.set_ylabel("Velocity")

ax = axes[2,1]
ax.scatter(sod_sc["position-x"], vel_sc, color="steelblue", label="single-core")
ax.plot(pos_ex, vel_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel("X")
ax.set_ylabel("Velocity")

# sort by global ids
ids_sc = np.argsort(sod_sc["ids"])
ids_mc = np.argsort(sod_mc["ids"])

ax = axes[1,2]
ax.scatter(sod_sc["density"][ids_sc], sod_mc["density"][ids_mc], marker="x", color="indianred")
ax.plot([0, 5], [0, 5], color="k")
ax.set_xlim(0,1.1)
ax.set_ylim(0,1.1)
ax.set_xlabel("Density (SC)")
ax.set_ylabel("Density (MC)")

ax = axes[2,2]
ax.scatter(vel_sc[ids_sc], vel_mc[ids_mc], marker="x", color="indianred")
ax.plot([-1, 2], [-1, 2], color="k")
ax.set_xlim(-0.1,1.1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel("Velocity (SC)")
ax.set_ylabel("Velocity (MC)")

ax = axes[0,2]
ax.scatter(np.log10(sod_sc["volume"][ids_sc]), np.log10(sod_mc["volume"][ids_mc]), marker="x", color="indianred")
ax.plot([-5, -1], [-5, -1], color="k")
ax.set_xlim(-3.8,-3.4)
ax.set_ylim(-3.8,-3.4)
ax.set_xlabel("log(Volume) (SC)")
ax.set_ylabel("log(Volume) (MC)")

plt.tight_layout()
plt.savefig("compare_single_multi_cartesian.pdf")
plt.show()
