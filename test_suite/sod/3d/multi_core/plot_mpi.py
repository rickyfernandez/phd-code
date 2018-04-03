import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

# stitch back multi-core solution
num_proc = 4
io = phd.Hdf5()
base_name = "mpi_sod_output/final_output/final_output0000"
for i in range(num_proc):

    cpu = "_cpu" + str(i).zfill(4)
    file_name = base_name + cpu + "/final_output0000" + cpu + ".hdf5"

    if i == 0:
        sod = io.read(file_name)
    else:
        sod.append_container(io.read(file_name))
sod.remove_tagged_particles(phd.ParticleTAGS.Ghost)

# exact riemann solution
f = h5py.File("riemann_sol.hdf5", "r")
pos_ex = f["/x"][:]
rho_ex = f["/density"][:]
pre_ex = f["/pressure"][:]
vel_ex = f["/velocity"][:]
f.close()

fig, axes = plt.subplots(3,1, figsize=(6,12))
plt.suptitle("Sod Simulation")

ax = axes[0]
ax.scatter(sod["position-x"], sod["density"], color="steelblue", label="simulation")
ax.plot(pos_ex, rho_ex, "k", label="exact")
ax.set_xlim(0,1.0)
ax.set_ylim(0,1.1)
ax.set_xlabel("X")
ax.set_ylabel("Density")
ax.legend()

ax = axes[1]
vel = np.sqrt(sod["velocity-x"]**2 + sod["velocity-y"]**2)
ax.scatter(sod["position-x"], vel, color="steelblue")
ax.plot(pos_ex, vel_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel("X")
ax.set_ylabel("Velocity")

ax = axes[2]
ax.scatter(sod["position-x"], sod["pressure"], color="steelblue")
ax.plot(pos_ex, pre_ex, "k")
ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel("X")
ax.set_ylabel("Pressure")

plt.savefig("sod_3d_mpi.png")
plt.show()
