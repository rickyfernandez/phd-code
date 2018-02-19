import phd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mat_colors
from matplotlib.collections import PatchCollection

# single-core solution
file_name="../single_core/random/sedov_output/final_output/final_output0000/final_output0000.hdf5"
reader = phd.Hdf5()
sedov = reader.read(file_name)

# exact sedov solution
exact  = np.loadtxt("exact_sedov_2d.dat")
rad_ex = exact[:,1]
rho_ex = exact[:,2]
pre_ex = exact[:,4]
vel_ex = exact[:,5]

fig, axes = plt.subplots(2,2, figsize=(12,12))
plt.suptitle("Sedov Simulation")

patch, colors = phd.vor_collection(sedov, "density")
sedov.remove_tagged_particles(phd.ParticleTAGS.Ghost)

p = PatchCollection(patch, alpha=0.4)
p.set_array(np.array(colors))
p.set_clim([0, 4.0])
ax = axes[0,0]
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

# put position and velocity in radial coordinates
rad = np.sqrt((sedov["position-x"]-0.5)**2 + (sedov["position-y"]-0.5)**2)
vel = np.sqrt(sedov["velocity-x"]**2 + sedov["velocity-y"]**2)

ax = axes[0,1]
ax.scatter(rad, sedov["density"], color="steelblue", label="simulation")
ax.plot(rad_ex, rho_ex, "k", label="exact")
ax.set_xlim(0,0.5)
ax.set_ylim(-1,7)
ax.set_xlabel("Radius")
ax.set_ylabel("Density")
ax.legend(loc="upper left")

ax = axes[1,0]
ax.scatter(rad, vel, color="steelblue")
ax.plot(rad_ex, vel_ex, "k")
ax.set_xlim(0,0.5)
ax.set_ylim(-0.5,2)
ax.set_xlabel("Radius")
ax.set_ylabel("Velocity")

ax = axes[1,1]
ax.scatter(rad, sedov["pressure"], color="steelblue")
ax.plot(rad_ex, pre_ex, "k")
ax.set_xlim(0,0.5)
ax.set_ylim(-0.5,2.5)
ax.set_xlabel("Radius")
ax.set_ylabel("Pressure")

plt.savefig("sedov_2d_single_core_random.pdf")
plt.show()
