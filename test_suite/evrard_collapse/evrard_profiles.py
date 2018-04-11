import phd
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16

def radial_profile(particles, field):

    # create radius of particles
    radius = np.sqrt((particles["position-x"]-1.25)**2 +\
            (particles["position-y"]-1.25)**2 +\
            (particles["position-z"]-1.25)**2)

    # create bin radius
    rmin, rmax = min(radius), max(radius)
    radial_bins = np.logspace(-3, 1)

    dr = radial_bins[1:] - radial_bins[0:-1]

    dens = np.zeros(radial_bins.size)
    coun = np.zeros(radial_bins.size)
    for i in range(particles.get_carray_size()):
        index = 0

        if radius[i] < 1.e-3 or radius[i] > 1.:
            continue

        while index+1 < len(radial_bins) and radius[i] > radial_bins[index+1]:
            index += 1
        dens[index] += particles[field][i]
        coun[index] += 1.

    for i in range(len(radial_bins)):
        if coun[i] > 0:
            dens[i] /= coun[i] 

    return radial_bins[0:-1] + 0.5*dr, dens[0:-1] 


exact = np.loadtxt("exact/profile3d027.txt")

# initial output
io = phd.Hdf5()
#file_name = "evrard2_output/time_interval/time_interval0080/time_interval0080.hdf5"
#file_name = "test_collapse_output/final_output/final_output0000/final_output0000.hdf5"
#file_name = "blnr_output/time_interval/time_interval0080/time_interval0080.hdf5"
file_name = "evrard_output/final_output/final_output0000/final_output0000.hdf5"
particles = io.read(file_name)
particles.remove_tagged_particles(phd.ParticleTAGS.Ghost)

time = h5py.File(file_name, "r").attrs["time"]
print time

r = np.sqrt((particles["position-x"]-1.25)**2 + (particles["position-y"]-1.25)**2 + (particles["position-z"]-1.25)**2)
fig, axes = plt.subplots(1,3, figsize=(12,6))
#fig.suptitle("Evrard Collapse Problem: time=%0.2f" % time, fontsize=14)

num_particles = particles.get_carray_size()
particles.register_carray(num_particles, "velocity", "double")
particles.register_carray(num_particles, "entropic", "double")

particles["velocity"][:] = (particles["velocity-x"]*(particles["position-x"]-1.25) +\
        particles["velocity-y"]*(particles["position-y"]-1.25) +\
        particles["velocity-z"]*(particles["position-z"]-1.25))/r
particles["entropic"][:] = particles["pressure"]/particles["density"]**(5/3.)

# create new fields

rad_den, den = radial_profile(particles, "density")
rad_vel, vel = radial_profile(particles, "velocity")
rad_ent, ent = radial_profile(particles, "entropic")

ax = axes[0]
ax.loglog(rad_den, den, 'o', c="steelblue")
ax.loglog(exact[:,0], exact[:,1], "r-")
ax.set_xlim(1.0E-2, 0.8)
ax.set_ylim(1.0E-2, 1.0E3)
ax.set_xlabel(r"$r$", fontsize=18)
ax.set_ylabel(r"$\rho$", fontsize=18)
ax.tick_params(direction="in", right=True, top=True, which="both")

ax = axes[1]
ax.semilogx(rad_vel, vel, 'o', c="steelblue")
ax.semilogx(exact[:,0], exact[:,2], "r-")
ax.set_xlim(1.0E-2, 0.8)
ax.set_ylim(-1.8, .1)
ax.set_xlabel(r"$r$", fontsize=18)
ax.set_ylabel(r"$V_r$", fontsize=18)
ax.tick_params(direction="in", right=True, top=True, which="both")


ax = axes[2]
ax.semilogx(rad_ent, ent, 'o', c="steelblue")
ax.semilogx(exact[:,0], exact[:,3]/exact[:,1]**(5./3.), "r-")
ax.set_xlim(1.0E-2, 0.8)
ax.set_ylim(0, .3)
ax.set_xlabel(r"$r$", fontsize=18)
ax.set_ylabel(r"$P/\rho^\gamma$", fontsize=18)
ax.tick_params(direction="in", right=True, top=True, which="both")
fig.tight_layout()
#fig.subplots_adjust(top=0.9)
#plt.savefig("evrard.eps")
plt.show()
