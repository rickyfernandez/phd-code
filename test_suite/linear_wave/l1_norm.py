import phd
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from sklearn.linear_model import LinearRegression

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16

l1 = np.zeros(5)
resolution = np.array([10, 20, 40, 80, 160])
for i, res in enumerate(resolution):

    file_name = "linear_wave_" + str(res) + "_output/final_output/final_output0000/final_output0000.hdf5"
    reader = phd.Hdf5()
    particles = reader.read(file_name)
    particles.remove_tagged_particles(phd.ParticleTAGS.Ghost)

    for j in range(particles.get_carray_size()):
        l1[i] += np.abs(particles["density"][j] -\
                (1.0 + 1.0e-6*np.sin(2*np.pi*particles["position-x"][j])))

    l1[i] = l1[i]/particles.get_carray_size()

lr = LinearRegression()
lr.fit(np.log10(resolution.reshape(-1,1)), np.log10(l1))

x = np.log10(np.linspace(7, 180).reshape(-1,1))
y = lr.predict(x)

fig, ax = plt.subplots(1,1, figsize=(6,6))
ax.set_title("L1 Norm of Linear Wave", fontsize=18, y=1.02)

ax.loglog(resolution, l1, "o", color="steelblue")
ax.loglog(10**x.flatten(), 10**y, "r", label=r"$\sim N^{%0.2f}$"%lr.coef_[0])
ax.set_xlabel(r"$N$", fontsize=18)
ax.set_ylabel("L1", fontsize=18)
ax.legend(fontsize=18)

fig.tight_layout()
plt.savefig("linear-wave-l1.eps")
plt.show()
