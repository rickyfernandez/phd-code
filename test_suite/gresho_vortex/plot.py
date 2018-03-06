import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mat_colors
from matplotlib.collections import PatchCollection

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
        "gv_output/iteration_interval/iteration_interval0001/iteration_interval0001.hdf5",
        "gv_output/iteration_interval/iteration_interval0011/iteration_interval0011.hdf5",
        "gv_output/final_output/final_output0000/final_output0000.hdf5"
        ]

fig, axes = plt.subplots(2,3, figsize=(12,6))
plt.suptitle("Gresho Vortex Simulation")

for i, fi in enumerate(files):

    reader = phd.Hdf5()
    vortex = reader.read(fi)

    time = h5py.File(fi, "r").attrs["time"]

    # third column
    patch, colors = phd.vor_collection(vortex, "density")
    vortex.remove_tagged_particles(phd.ParticleTAGS.Ghost)

    p = PatchCollection(patch, edgecolor="black", alpha=0.4)
    p.set_array(np.array(colors))
    p.set_clim([0, 4.0])
    ax = axes[0,i]
    ax.set_title("Time=%0.2f" % time)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.add_collection(p)

    # put position and velocity in radial coordinates
    rad = np.sqrt((vortex["position-x"]-0.5)**2 + (vortex["position-y"]-0.5)**2)
    vel = np.sqrt(vortex["velocity-x"]**2 + vortex["velocity-y"]**2)

    ax = axes[1,i]
    ax.scatter(rad, vel, color="steelblue", label="simulation")
    ax.plot(radius, sol, "k--", lw=0.8, label="exact")
    ax.set_xlim(0,0.5)
    ax.set_ylim(0,1)

#plt.savefig("sedov_2d_single_core_random.pdf")
plt.show()
