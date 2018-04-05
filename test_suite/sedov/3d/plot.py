import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt

# plot cartesian or uniform run
file_name="sedov_output/final_output/final_output0000/final_output0000.hdf5"
reader = phd.Hdf5()
sedov = reader.read(file_name)
sedov.remove_tagged_particles(phd.ParticleTAGS.Ghost)

r = np.sqrt((sedov["position-x"]-0.5)**2 + (sedov["position-y"]-0.5)**2 + (sedov["position-z"]-0.5)**2)
v = np.sqrt(sedov["velocity-x"]**2 + sedov["velocity-y"]**2 + sedov["velocity-z"]**2)

# get the exact solution
exact = np.loadtxt("sedov_3d.dat")

# get the exact solution
x_ex = exact[:,1]   # radius
r_ex = exact[:,2]   # density
p_ex = exact[:,4]   # pressure
u_ex = exact[:,5]   # velocity

plt.figure(figsize=(6,12))
plt.subplot(3,1,1)
plt.scatter(r, sedov["density"], color="lightsteelblue", label="phd")
plt.plot(x_ex, r_ex, "k", label="exact")
plt.xlim(0,0.5)
plt.ylim(-1,4.1)
plt.xlabel("Radius")
plt.ylabel("Density")
l = plt.legend(loc="upper left", prop={"size":12})
l.draw_frame(False)

plt.subplot(3,1,2)
plt.scatter(r, v, color="lightsteelblue")
plt.plot(x_ex, u_ex, "k")
plt.xlim(0,0.5)
plt.ylim(-0.5,2.0)
plt.xlabel("Radius")
plt.ylabel("Velocity")

plt.subplot(3,1,3)
plt.scatter(r, sedov["pressure"], color="lightsteelblue")
plt.plot(x_ex, p_ex, "k")
plt.xlim(0,0.5)
plt.ylim(-0.5,4.5)
plt.xlabel("Radius")
plt.ylabel("Pressure")

plt.tight_layout()
plt.savefig("sedov_3d.png")
plt.show()
