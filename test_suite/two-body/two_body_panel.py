import phd
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16

fig, axes = plt.subplots(1,2, figsize=(12,6))
reader = phd.Hdf5()

num = 1000
ax = axes[0]
time = np.zeros(num)
potential = np.zeros(num)
kinetic = np.zeros(num)
energy = np.zeros(num)
for i in range(num):

    if i == 0:
        file_name="two_body_output/initial_output/initial_output0000/initial_output0000.hdf5"
    else:
        file_name="two_body_output/iteration_interval/iteration_interval"+str(i).zfill(4)
        file_name=file_name+"/iteration_interval"+str(i).zfill(4)+".hdf5"

    particles = reader.read(file_name)
    ax.plot(particles["position-x"][0], particles["position-y"][0], ".k")
    ax.plot(particles["position-x"][1], particles["position-y"][1], ".k")

    v2 = particles["velocity-x"]**2 + particles["velocity-y"]**2
    kinetic[i] = (0.5*particles["mass"]*v2).sum()
    potential[i] = 0.5*(particles["mass"]*particles["potential"]).sum()
    energy[i] = kinetic[i] + potential[i]

    if i == 0:
        E0 = energy[i]

    time[i] = h5py.File(file_name, "r").attrs["time"]

error = (energy-E0)/E0
print "energy min:", error.min(), "energy max:", error.max()
ax.set_xlabel("X")
ax.set_ylabel("Y")

a = 0.5
c = 0.25
e = c/a

G = 1.0
m1 = 1.0
m2 = 2.0
q = m1/m2 
m = m1 + m2
T0 = np.sqrt(4.*np.pi**2*a**3/(G*m))
theta = np.linspace(0, 2*np.pi)
r = a*(1-e**2)/(1.-e*np.cos(theta))

x = r*np.cos(theta)
y = r*np.sin(theta)

x1 = m1*x/m
y1 = m1*y/m
x2 = -m2*x/m
y2 = -m2*y/m

# plot true solution
ax.plot(x1, y1, "r")
ax.plot(x2, y2, "r")
ax.set_xlim(-.6, .6)
ax.set_ylim(-.6, .6)
ax.set_aspect("equal")

ax = axes[1]
ax.plot(time/T0, error, c="red")
ax.set_xlabel(r"$t/T$", fontsize=18)
ax.set_ylabel(r"$\Delta E/E_{\mathrm{tot}}$", fontsize=18)

plt.tight_layout()
plt.savefig("two_body.eps")
plt.show()
