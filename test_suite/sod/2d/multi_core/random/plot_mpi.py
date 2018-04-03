import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

# stitch back multi-core solution
num_proc = 4
io = phd.Hdf5()
base_name = "sod_mpi_output/final_output/final_output0000"
for i in range(num_proc):

    cpu = "_cpu" + str(i).zfill(4)
    file_name = base_name + cpu + "/final_output0000" + cpu + ".hdf5"

    if i == 0:
        sod = io.read(file_name)
    else:
        sod.append_container(io.read(file_name))
tag = np.where(sod["type"] == phd.ParticleTAGS.Interior)[0]
sod.remove_items(tag)

# exact riemann solution
f = h5py.File("riemann_sol.hdf5", "r")
pos_ex = f["/x"][:]
rho_ex = f["/density"][:]
pre_ex = f["/pressure"][:]
vel_ex = f["/velocity"][:]
f.close()

fig, axes = plt.subplots(2,2, figsize=(12,12))
plt.suptitle("Sod Simulation")

patch, colors = phd.vor_collection(sod, "density")
sod.remove_tagged_particles(phd.ParticleTAGS.Ghost)

p = PatchCollection(patch, edgecolor="black", linewidth=0.3, alpha=0.8)
p.set_array(np.array(colors))
p.set_clim([0, 1.0])
ax = axes[0,0]
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

ax = axes[0,1]
ax.scatter(sod["position-x"], sod["density"], color="steelblue", label="simulation")
ax.plot(pos_ex, rho_ex, "k", label="exact")
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.1)
ax.set_xlabel("X")
ax.set_ylabel("Density")
ax.legend()

ax = axes[1,0]
vel = np.sqrt(sod["velocity-x"]**2 + sod["velocity-y"]**2)
ax.scatter(sod["position-x"], vel, color="steelblue")
ax.plot(pos_ex, vel_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel("X")
ax.set_ylabel("Velocity")

ax = axes[1,1]
ax.scatter(sod["position-x"], sod["pressure"], color="steelblue")
ax.plot(pos_ex, pre_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel("X")
ax.set_ylabel("Pressure")

plt.savefig("sod_2d_mpi.png")
plt.show()
