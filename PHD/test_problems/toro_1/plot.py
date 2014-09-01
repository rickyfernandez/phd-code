import h5py
import matplotlib.pyplot as plt

f1 = h5py.File("sod_hllc.hdf5", "r")
f2 = h5py.File("sod_hll.hdf5", "r")
f3 = h5py.File("sod_exact.hdf5", "r")
fe = h5py.File("riemann_sol.hdf5", "r")

num = f1["/density"].size

plt.figure(figsize=(8,8))

plt.subplot(3,1,1)
plt.scatter(f1["/particles"][0,:num], f1["/density"][:], facecolors='none', edgecolors='b', label="Hllc")
plt.scatter(f2["/particles"][0,:num], f2["/density"][:], facecolors='none', edgecolors='c', label="Hll")
plt.scatter(f3["/particles"][0,:num], f3["/density"][:], facecolors='none', edgecolors='r', label="Exact")
plt.plot(fe["/x"][:], fe["/density"], "k")
plt.ylabel("Density")
plt.xlim(0,1)
plt.ylim(0,1.1)
plt.title("Linear Reconstruction, Time: %0.2f" % f1.attrs["time"], fontsize=12)
l = plt.legend(loc="lower left", prop={"size":12})
l.draw_frame(False)

plt.subplot(3,1,2)
plt.scatter(f1["/particles"][0,:num], f1["/velocity-x"][:], facecolors='none', edgecolors='b')
plt.scatter(f2["/particles"][0,:num], f2["/velocity-x"][:], facecolors='none', edgecolors='c')
plt.scatter(f3["/particles"][0,:num], f3["/velocity-x"][:], facecolors='none', edgecolors='r')
plt.plot(fe["/x"][:], fe["/velocity"], "k")
plt.xticks([])
plt.ylabel("Velocity")
plt.xlim(0,1)
plt.ylim(-0.1,1.1)

plt.subplot(3,1,3)
plt.scatter(f1["/particles"][0,:num], f1["/pressure"][:], facecolors='none', edgecolors='b')
plt.scatter(f2["/particles"][0,:num], f2["/pressure"][:], facecolors='none', edgecolors='c')
plt.scatter(f3["/particles"][0,:num], f3["/pressure"][:], facecolors='none', edgecolors='r')
plt.plot(fe["/x"][:], fe["/pressure"], "k")
plt.xlabel("Position")
plt.ylabel("Pressure")
plt.xlim(0,1)
plt.ylim(0,1.1)

plt.tight_layout()
plt.savefig("toro_1")
plt.show()
