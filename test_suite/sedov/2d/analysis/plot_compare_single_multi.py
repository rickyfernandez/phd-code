import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

io = phd.Hdf5()

# single-core solution
base_name = "random/sedov_output/final_output/final_output0000"
file_name = "../single_core/" + base_name + "/final_output0000.hdf5"
sedov_sc = io.read(file_name)

# stitch back multi-core solution
# stitch back multi-core solution
num_proc = 4
for i in range(num_proc):

    cpu = "_cpu" + str(i).zfill(4)
    file_name = "../multi_core/" + base_name + cpu + "/final_output0000" + cpu + ".hdf5"

    if i == 0:
        sedov_mc = io.read(file_name)
    else:
        sedov_mc.append_container(io.read(file_name))

# exact sedov solution
exact  = np.loadtxt("exact_sedov_2d.dat")
rad_ex = exact[:,1]
rho_ex = exact[:,2]
per_ex = exact[:,4]
vel_ex = exact[:,5]

fig, axes = plt.subplots(3,3, figsize=(12,12))
patch, colors = phd.vor_collection(sedov_sc, "density")
sedov_sc.remove_tagged_particles(phd.ParticleTAGS.Ghost)

p = PatchCollection(patch, alpha=0.8)
p.set_array(np.array(colors))
p.set_clim([0, 6.0])
ax = axes[0,1]
ax.set_title("Single Core")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

# first remove interior particles
tag = np.where(sedov_mc["type"] == phd.ParticleTAGS.Interior)[0]
sedov_mc.remove_items(tag)

patch, colors = phd.vor_collection(sedov_mc, "process")
sedov_mc.remove_tagged_particles(phd.ParticleTAGS.Ghost)
p = PatchCollection(patch, alpha=0.8)
p.set_array(np.array(colors))

ax = axes[0,0]
ax.set_title("Multi Core")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

# put position and velocity in radial coordinates
rad_sc = np.sqrt((sedov_sc["position-x"]-0.5)**2 + (sedov_sc["position-y"]-0.5)**2)
rad_mc = np.sqrt((sedov_mc["position-x"]-0.5)**2 + (sedov_mc["position-y"]-0.5)**2)
vel_sc = np.sqrt(sedov_sc["velocity-x"]**2 + sedov_sc["velocity-y"]**2)
vel_mc = np.sqrt(sedov_mc["velocity-x"]**2 + sedov_mc["velocity-y"]**2)

# plot density curves
ax = axes[1,0]
ax.scatter(rad_mc, sedov_mc["density"], color="darkgray", label="multi-core")
ax.plot(rad_ex, rho_ex, "k")
ax.set_xlim(0,0.5)
ax.set_ylim(-1,7)
ax.set_xlabel("Radius")
ax.set_ylabel("Density")

ax = axes[1,1]
ax.scatter(rad_sc, sedov_sc["density"], color="steelblue", label="single-core")
ax.plot(rad_ex, rho_ex, "k")
ax.set_xlim(0,0.5)
ax.set_ylim(-1,7)
ax.set_xlabel("Radius")
ax.set_ylabel("Density")

ax = axes[2,0]
ax.scatter(rad_mc, vel_mc, color="darkgray", label="multi-core")
ax.plot(rad_ex, vel_ex, "k")
ax.set_xlim(0,0.5)
ax.set_ylim(-0.5,2)
ax.set_xlabel("Radius")
ax.set_ylabel("Velocity")

ax = axes[2,1]
ax.scatter(rad_sc, vel_sc, color="steelblue", label="single-core")
ax.plot(rad_ex, vel_ex, "k")
ax.set_xlim(0,0.5)
ax.set_ylim(-0.5,2)
ax.set_xlabel("Radius")
ax.set_ylabel("Velocity")

# sort by global ids
ids_sc = np.argsort(sedov_sc["ids"])
ids_mc = np.argsort(sedov_mc["ids"])

ax = axes[1,2]
ax.scatter(sedov_sc["density"][ids_sc], sedov_mc["density"][ids_mc], marker="x", color="indianred")
ax.plot([0, 5], [0, 5], color="k")
ax.set_xlim(0,4)
ax.set_ylim(0,4)
ax.set_xlabel("Density (SC)")
ax.set_ylabel("Density (MC)")

ax = axes[2,2]
ax.scatter(vel_sc[ids_sc], vel_mc[ids_mc], marker="x", color="indianred")
ax.plot([0, 2], [0, 2], color="k")
ax.set_xlim(0,1.5)
ax.set_ylim(0,1.5)
ax.set_xlabel("Velocity (SC)")
ax.set_ylabel("Velocity (MC)")

ax = axes[0,2]
ax.scatter(np.log10(sedov_sc["volume"][ids_sc]), np.log10(sedov_mc["volume"][ids_mc]), marker="x", color="indianred")
ax.plot([-5, -1], [-5, -1], color="k")
ax.set_xlim(-5,-1)
ax.set_ylim(-5,-1)
ax.set_xlabel("log(Volume) (SC)")
ax.set_ylabel("log(Volume) (MC)")

plt.tight_layout()
plt.savefig("compare_single_multi_random.pdf")
plt.show()
