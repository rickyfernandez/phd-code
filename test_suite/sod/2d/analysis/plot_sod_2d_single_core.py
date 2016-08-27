import phd
import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('../single_core/cartesian/sod_2d_cartesian_output/sod_2d_cartesian_0072.hdf5', 'r')
fe = h5py.File("riemann_sol.hdf5", "r")

indices = f['/tag'][:] == phd.ParticleTAGS.Real

plt.figure(figsize=(8,8))
plt.subplot(3,1,1)
plt.scatter(f["/position-x"][indices], f["/density"][indices], color='lightsteelblue', label="phd")
plt.plot(fe["/x"][:], fe["/density"], "k", label="Exact")
plt.xlim(0,1.0)
plt.ylim(0,1.1)
plt.ylabel("Density")
plt.title('Constant Reconstruction, Time=%0.2f, N=%d' % (f.attrs['time'], np.sum(indices)), fontsize=12)
l = plt.legend(loc="upper right", prop={"size":12})
l.draw_frame(False)

plt.subplot(3,1,2)
plt.scatter(f["/position-x"][indices], np.sqrt(f["/velocity-x"][indices]**2 + f["/velocity-y"][indices]**2),
        color='lightsteelblue')
plt.plot(fe["/x"][:], fe["/velocity"], "k", label="Exact")
plt.xlim(0,1)
plt.ylim(-0.1,1.1)
plt.ylabel("Velocity")

plt.subplot(3,1,3)
plt.scatter(f["/position-x"][indices], f["/pressure"][indices], color='lightsteelblue')
plt.plot(fe["/x"][:], fe["/pressure"], "k", label="Exact")
plt.xlim(0,1)
plt.ylim(0,1.1)
plt.xlabel("Position")
plt.ylabel("Pressure")

plt.tight_layout()
plt.savefig("sod_2d_single_core.pdf")
plt.show()
