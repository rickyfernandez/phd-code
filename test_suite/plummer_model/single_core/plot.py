import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plummer(r, M=1.):
    return 3.*M/4./np.pi*(1.+r**2)**(-5./2.)

def radial_profile(particles):

    # find center mass
    totmass = 0.
    com = [0., 0., 0.]
    for i in range(particles.get_carray_size()):
        for j, ax in enumerate("xyz"):
            com[j] += particles["mass"][i]*particles["position-"+ax][i]
        totmass += particles["mass"][i]
    for j in range(3):
        com[j] /= totmass

    # create radius of particles
    radius = np.zeros(particles.get_carray_size())
    for i in range(particles.get_carray_size()):
        radius[i] = np.sqrt(
                (particles["position-x"][i]-com[0])**2 +\
                (particles["position-y"][i]-com[1])**2 +\
                (particles["position-z"][i]-com[2])**2)

    # create bin radius
    rmin, rmax = min(radius), max(radius)
    dr = 0.01*(rmax - rmin)
    radial_bins = np.arange(rmin, rmax + dr, dr)

    dens = np.zeros(radial_bins.size)
    for i in range(particles.get_carray_size()):
        index = 0
        while index+1 < len(radial_bins) and radius[i] > radial_bins[index+1]:
            index += 1
        dens[index] += particles["mass"][i]

    for i in range(len(radial_bins)):
        dens[i] /= (4*np.pi/3.*(3*radial_bins[i]*dr**2 + 3.*radial_bins[i]**2*dr + dr**3))

    return radial_bins, dens 

# initial output
io = phd.Hdf5()
file_name = "plummer_output/initial_output/initial_output0000/initial_output0000.hdf5"
initial_time = h5py.File(file_name, "r").attrs["time"]
initial_particles = io.read(file_name)

# final output
io = phd.Hdf5()
file_name = "plummer_output/final_output/final_output0000/final_output0000.hdf5"
final_time = h5py.File(file_name, "r").attrs["time"]
final_particles = io.read(file_name)

initial_radial, initial_dens = radial_profile(initial_particles)
final_radial, final_dens = radial_profile(final_particles)

fig, ax = plt.subplots(1,1, figsize=(6,6))
r_pl = np.logspace(-2, 2)

ax.set_title("Plummer Sphere")
ax.loglog(r_pl, plummer(r_pl, 1000.))
ax.loglog(initial_radial, initial_dens, 'g^', label=r"t=0")
ax.loglog(final_radial, final_dens, "or", alpha=0.8, label="t = %0.3f" % final_time)
ax.set_xlim(1.0E-2, 30)
ax.set_ylim(1.0E-3, 1.0E3)
ax.set_xlabel("r")
ax.set_ylabel(r"$\rho$")
ax.legend()
plt.show()
