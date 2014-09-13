import h5py
import numpy as np
import matplotlib.pyplot as plt

f1 = h5py.File("sedov_hllc.hdf5", "r")
#f2 = h5py.File("sedov_hll.hdf5", "r")
f2 = h5py.File("sedov_exact.hdf5", "r")

# get the exact solution
exact = np.loadtxt("sedov_2d.dat")

x_exact   = exact[:,1]
rho_exact = exact[:,2]

# convert to radial
num = f1["/density"].size
x1 = f1["/particles"][0,:num]
y1 = f1["/particles"][1,:num]
r1 = np.sqrt((x1-0.5)**2 + (y1-0.5)**2)

num = f2["/density"].size
x2 = f2["/particles"][0,:num]
y2 = f2["/particles"][1,:num]
r2 = np.sqrt((x2-0.5)**2 + (y2-0.5)**2)

plt.figure(figsize=(6,6))

plt.scatter(r1, f1["/density"][:], facecolors='none', edgecolors='b', label="Hllc")
#plt.scatter(r2, f2["/density"][:], facecolors='none', edgecolors='c', label="Hll")
plt.scatter(r2, f2["/density"][:], facecolors='none', edgecolors='c', label="Exact")
plt.plot(x_exact, rho_exact, "r")
plt.xlim(0,0.5)
plt.ylim(0,7)
plt.xlabel("Position")
plt.ylabel("Density")
plt.title("Linear Reconstruction, Time: %0.2f" % f1.attrs["time"], fontsize=12)
l = plt.legend(loc="upper left", prop={"size":12})
l.draw_frame(False)
#plt.savefig("sedov")
plt.show()
