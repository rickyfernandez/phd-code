import h5py
import numpy as np
import matplotlib.pyplot as plt

f1 = h5py.File("sedov_hllc.hdf5", "r")
f2 = h5py.File("sedov_hll.hdf5", "r")
f3 = h5py.File("sedov_exact.hdf5", "r")

# get the exact solution
exact = np.loadtxt("sedov_2d.dat")

x_exact   = exact[:,1]
rho_exact = exact[:,2]
p_exact   = exact[:,4]
u_exact   = exact[:,5]

# convert to radial
num = f1["/density"].size
x1 = f1["/particles"][0,:num]
y1 = f1["/particles"][1,:num]
r1 = np.sqrt((x1-0.5)**2 + (y1-0.5)**2)

num = f2["/density"].size
x2 = f2["/particles"][0,:num]
y2 = f2["/particles"][1,:num]
r2 = np.sqrt((x2-0.5)**2 + (y2-0.5)**2)

num = f3["/density"].size
x3 = f3["/particles"][0,:num]
y3 = f3["/particles"][1,:num]
r3 = np.sqrt((x3-0.5)**2 + (y3-0.5)**2)

plt.figure(figsize=(8,8))

plt.subplot(3,1,1)
plt.scatter(r1, f1["/density"][:], facecolors='none', edgecolors='b', label="Hllc")
plt.scatter(r2, f2["/density"][:], facecolors='none', edgecolors='c', label="Hll")
plt.scatter(r3, f3["/density"][:], facecolors='none', edgecolors='r', label="Exact")
plt.plot(x_exact, rho_exact, "k")
plt.xlim(0,0.5)
plt.ylim(-1,7)
plt.xlabel("Position")
plt.ylabel("Density")
plt.title("Linear Reconstruction, Time: %0.2f" % f1.attrs["time"], fontsize=12)
l = plt.legend(loc="upper left", prop={"size":12})
l.draw_frame(False)

plt.subplot(3,1,2)
plt.scatter(r1, np.sqrt(f1["/velocity-x"][:]**2 + f1["/velocity-y"][:]**2), facecolors='none', edgecolors='b', label="Hllc")
plt.scatter(r2, np.sqrt(f2["/velocity-x"][:]**2 + f2["/velocity-y"][:]**2), facecolors='none', edgecolors='c', label="Hll")
plt.scatter(r3, np.sqrt(f3["/velocity-x"][:]**2 + f3["/velocity-y"][:]**2), facecolors='none', edgecolors='r', label="Exact")
plt.plot(x_exact, u_exact, "k")
plt.xlim(0,0.5)
plt.ylim(-0.5,2.0)
plt.ylabel("Velocity")

plt.subplot(3,1,3)
plt.scatter(r1, f1["/pressure"][:], facecolors='none', edgecolors='b', label="Hllc")
plt.scatter(r2, f2["/pressure"][:], facecolors='none', edgecolors='c', label="Hll")
plt.scatter(r3, f3["/pressure"][:], facecolors='none', edgecolors='r', label="Exact")
plt.plot(x_exact, p_exact, "k")
plt.xlim(0,0.5)
plt.ylim(-0.5,3.0)
plt.ylabel("Pressure")

plt.savefig("sedov.pdf")
plt.show()
