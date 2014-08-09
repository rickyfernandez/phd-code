import h5py
import matplotlib.pyplot as plt

fs = h5py.File("Sod.hdf5", "r")
fe = h5py.File("riemann_sol.hdf5", "r")

num = fs["/density"].size

plt.figure(figsize=(6,8))

plt.subplot(3,1,1)
plt.scatter(fs["/particles"][0,:num], fs["/density"][:], facecolors='none', edgecolors='b')
plt.plot(fe["/x"][:], fe["/density"], "r")
plt.ylabel("Density")
plt.xlim(0,1)
plt.ylim(0,1.1)
plt.title("time: %0.2f" % fs.attrs["time"])

plt.subplot(3,1,2)
plt.scatter(fs["/particles"][0,:num], fs["/velocity"][0,:], facecolors='none', edgecolors='b')
plt.plot(fe["/x"][:], fe["/velocity"], "r")
plt.xticks([])
plt.ylabel("Velocity")
plt.xlim(0,1)
plt.ylim(0,1.1)

plt.subplot(3,1,3)
plt.plot(fe["/x"][:], fe["/pressure"], "r")
plt.scatter(fs["/particles"][0,:num], fs["/pressure"][:], facecolors='none', edgecolors='b')
plt.xlabel("Position")
plt.ylabel("Pressure")
plt.xlim(0,1)
plt.ylim(0,1.1)

plt.tight_layout()
plt.savefig("Torro_01")
plt.show()
