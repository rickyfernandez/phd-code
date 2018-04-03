import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

# single-core solution
file_name1="linear_wave_80_output/initial_output/initial_output0000/initial_output0000.hdf5"
file_name2="linear_wave_80_output/final_output/final_output0000/final_output0000.hdf5"
reader = phd.Hdf5()
particles1 = reader.read(file_name1)
particles2 = reader.read(file_name2)

fig, axes = plt.subplots(2,2, figsize=(12,12))

patch, colors = phd.vor_collection(particles1, "density")
particles1.remove_tagged_particles(phd.ParticleTAGS.Ghost)
particles2.remove_tagged_particles(phd.ParticleTAGS.Ghost)

p = PatchCollection(patch, cmap="jet", edgecolor="black", alpha=0.4)
p.set_array(np.array(colors))
ax = axes[0,0]
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

ax = axes[0,1]
ax.plot(particles1["position-x"], particles1["density"], "kx", label="t=0.")
ax.plot(particles2["position-x"], particles2["density"], "r.", label="t=1.")
ax.set_xlim(0,1.0)
ax.set_ylim(1.0-2.0e-6,1.0+2.0e-6)
ax.set_xlabel("X")
ax.set_ylabel("Density")
ax.legend()

ax = axes[1,0]
ax.plot(particles1["position-x"], particles1["velocity-x"], "kx", label="t=0.")
ax.plot(particles2["position-x"], particles2["velocity-x"], "r.", label="t=1.")
ax.set_xlim(0,1)
ax.set_ylim(-2.0e-6,+2.0e-6)
ax.set_xlabel("X")
ax.set_ylabel("Velocity")

ax = axes[1,1]
ax.plot(particles1["position-x"], particles1["pressure"], "kx", label="t=0")
ax.plot(particles2["position-x"], particles2["pressure"], "r.", label="t=1")
ax.set_xlim(0,1)
ax.set_ylim(3./5.-2.0e-6,3.0/5.+2.0e-6)
ax.set_xlabel("X")
ax.set_ylabel("Pressure")

plt.tight_layout()
plt.savefig("linear_wave_plot.png")
plt.show()
