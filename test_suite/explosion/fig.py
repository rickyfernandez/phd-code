import phd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16

# single-core solution
file_name="explosion_output/final_output/final_output0000/final_output0000.hdf5"
reader = phd.Hdf5()
sedov = reader.read(file_name)

# exact sedov solution
exact  = np.loadtxt("solution.txt")
rad_ex = exact[:,0]
rho_ex = exact[:,1]
vel_ex = exact[:,2]
pre_ex = exact[:,3]

fig, axes = plt.subplots(1,2, figsize=(12,6))

patch, colors = phd.vor_collection(sedov, "density")
sedov.remove_tagged_particles(phd.ParticleTAGS.Ghost)

p = PatchCollection(patch, edgecolor="none", linewidth=0.1, alpha=0.8)
p.set_array(np.array(colors))
p.set_clim([0, 1.0])
ax = axes[0]
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.add_collection(p)

# put position and velocity in radial coordinates
rad = np.sqrt((sedov["position-x"]-0.5)**2 + (sedov["position-y"]-0.5)**2)
#vel = np.sqrt(sedov["velocity-x"]**2 + sedov["velocity-y"]**2)
vel = (sedov["velocity-x"]*(sedov["position-x"]-0.5) +\
        sedov["velocity-y"]*(sedov["position-y"]-0.5))/rad

ax = axes[1]
ax.plot(rad, sedov["density"], ".", color="steelblue", label="simulation")
ax.plot(rad_ex, rho_ex, "red", label="exact")
ax.set_xlim(0,0.75)
ax.set_ylim(0,1.1)
ax.set_xlabel(r"$r$", fontsize=18)
ax.set_ylabel(r"$\rho$", fontsize=18)

#ax = axes[1,0]
#ax.plot(rad, vel, ".", color="steelblue")
#ax.plot(rad_ex, vel_ex, "k")
#ax.set_xlim(0,0.75)
#ax.set_ylim(-0.1,1.2)
#ax.set_xlabel(r"$r$", fontsize=18)
#ax.set_ylabel(r"$v_r$", fontsize=18)
#
#ax = axes[1,1]
#ax.plot(rad, sedov["pressure"], ".", color="steelblue")
#ax.plot(rad_ex, pre_ex, "k")
#ax.set_xlim(0,0.75)
#ax.set_ylim(0.,1.1)
#ax.set_xlabel(r"$r$", fontsize=18)
#ax.set_ylabel(r"$P$", fontsize=18)

plt.tight_layout()
plt.savefig("explosion_2d.eps")
plt.show()
