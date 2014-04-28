from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Polygon
import moving_mesh.mesh as mesh
import matplotlib.pyplot as plt
import numpy as np

def sedov():

    # boundaries
    boundary_dic = {"left":0.0, "right":1.0, "bottom":0.0, "top":1.0}

    L = 1.       # domain size
    n = 50      # number of points
    gamma = 1.4

    dx = L/n
    x = (np.arange(n+6, dtype=np.float64) - 3)*dx + 0.5*dx
    X, Y = np.meshgrid(x,x); Y = np.flipud(Y)

    #X[0::2, :] += 0.5*dx
    #X[1::2, :] -= 0.25*dx

    x = X.flatten(); y = Y.flatten()

    left   = boundary_dic["left"];   right = boundary_dic["right"]
    bottom = boundary_dic["bottom"]; top   = boundary_dic["top"]


    indices = (((left < x) & (x < right)) & ((bottom < y) & (y < top)))
    x_in = x[indices]; y_in = y[indices]

    data = np.zeros((4, x_in.size))

    data[0,:] = 1.0                    # density
    data[3,:] = 1.0E-5                 # energy density

    r = 0.0125/2.
    cells = ((x_in-.5)**2 + (y_in-.5)**2) <= r
    data[3, cells] = 1.0/(np.pi*r**2)

    
    x_particles = np.copy(x_in); y_particles = np.copy(y_in)
    particles_index = {}
    particles_index["real"] = np.arange(x_particles.size)

    x_particles = np.append(x_particles, x[~indices])
    y_particles = np.append(y_particles, y[~indices])
    particles_index["ghost"] = np.arange(particles_index["real"].size, x_particles.size)

    particles = np.array(zip(x_particles, y_particles))

    return data, particles, gamma, particles_index, boundary_dic

if __name__ == "__main__":

    data, particles, _, particles_index, boundary_dic = Sedov()

    neighbor_graph, face_graph, circum_centers = mesh.tessellation(particles)

    l = []
    for i in particles_index["real"]:
        verts = circum_centers[np.asarray(face_graph[i])[:,0]]
        l.append(Polygon(verts, True))

    # add colormap
    colors = []
    for i in particles_index["real"]:
        colors.append(data[3,i])

    fig, ax = plt.subplots()
    p = PatchCollection(l, cmap=matplotlib.cm.jet, alpha=0.4)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    plt.colorbar(p)
    plt.show()
