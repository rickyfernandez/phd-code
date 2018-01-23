import phd
import numpy as np
import matplotlib.pyplot as plt


# single-core solution
file_name="sedov_output/final_output/final_output0000/final_output0000.hdf5"
reader = phd.Hdf5()
sedov = reader.read(file_name)

indices = sedov["tag"][:] == phd.ParticleTAGS.Real
x = sedov["position-x"][indices]
y = sedov["position-y"][indices]
z = sedov["position-z"][indices]
r = np.sqrt((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)
v = np.sqrt(sedov["velocity-x"][indices]**2 + sedov["velocity-y"][indices]**2 + sedov["velocity-z"][indices]**2)

# get the exact solution
exact = np.loadtxt("exact_sedov_3d.dat")

# get the exact solution
x_ex = exact[:,1]   # radius
r_ex = exact[:,2]   # density
p_ex = exact[:,4]   # pressure
u_ex = exact[:,5]   # velocity

plt.figure(figsize=(8,8))
plt.subplot(3,1,1)
plt.scatter(r, sedov["density"][indices], color="lightsteelblue", label="phd")
plt.plot(x_ex, r_ex, "k", label="exact")
plt.xlim(0,0.8)
plt.ylim(-1,7)
plt.ylabel("Density")
l = plt.legend(loc="upper left", prop={"size":12})
l.draw_frame(False)

plt.subplot(3,1,2)
plt.scatter(r, v, color="lightsteelblue")
plt.plot(x_ex, u_ex, "k")
plt.xlim(0,0.8)
plt.ylim(-0.5,2.0)
plt.ylabel("Velocity")

plt.subplot(3,1,3)
plt.scatter(r, sedov["pressure"][indices], color="lightsteelblue")
plt.plot(x_ex, p_ex, "k")
plt.xlim(0,0.8)
plt.ylim(-0.5,3.0)
plt.xlabel("Position")
plt.ylabel("Pressure")

plt.tight_layout()
plt.savefig("sedov_3d_uniform_single_core.png")
plt.show()
