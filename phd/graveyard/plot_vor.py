# for debug plotting 
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib

def vor_plot2(pc, mesh, iteration_count):

    # debugging plot --- turn to a routine later ---
    l = []
    ii = 0; jj = 0
    for i in range(pc.get_number_of_particles()):

        jj += mesh['number of neighbors'][i]*2

        if pc['tag'][i] == ParticleTAGS.Real or pc['type'][i] == ParticleTAGS.Boundary:

            verts_indices = np.unique(mesh['faces'][ii:jj])
            verts = mesh['voronoi vertices'][verts_indices]

            # coordinates of neighbors relative to particle p
            xc = verts[:,0] - pc['position-x'][i]
            yc = verts[:,1] - pc['position-y'][i]

            # sort in counter clock wise order
            sorted_vertices = np.argsort(np.angle(xc+1j*yc))
            verts = verts[sorted_vertices]

            l.append(Polygon(verts, True))

        ii = jj

    dens = pc['density']

    # add colormap
    colors = []
    for i in range(pc.get_number_of_particles()):
        if pc['tag'][i] == ParticleTAGS.Real or pc['type'][i] == ParticleTAGS.Boundary:
            colors.append(dens[i])

    fig, ax = plt.subplots()
    p = PatchCollection(l, alpha=0.4)
    p.set_array(np.array(colors))
    p.set_clim([0, 4])

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.add_collection(p)
    ax.set_aspect('equal')

    plt.colorbar(p)
    plt.savefig('test_'+ `iteration_count`.zfill(4) + '.pdf', format='pdf')

    plt.cla()
    plt.clf()

##def vor_plot(pc, mesh, rank, load_balance):
def vor_plot(pc, mesh, rank):

    # debugging plot --- turn to a routine later ---
    l = []
    ii = 0; jj = 0
    for i in range(pc.get_number_of_particles()):

        jj += mesh['number of neighbors'][i]*2

        if pc['tag'][i] == ParticleTAGS.Real or pc['type'][i] == ParticleTAGS.Boundary or pc['type'][i] == ParticleTAGS.BoundarySecond:

            verts_indices = np.unique(mesh['faces'][ii:jj])
            verts = mesh['voronoi vertices'][verts_indices]

            # coordinates of neighbors relative to particle p
            xc = verts[:,0] - pc['position-x'][i]
            yc = verts[:,1] - pc['position-y'][i]

            # sort in counter clock wise order
            sorted_vertices = np.argsort(np.angle(xc+1j*yc))
            verts = verts[sorted_vertices]

            l.append(Polygon(verts, True))

        ii = jj

    colors = []
    for i in range(pc.get_number_of_particles()):
        if pc['tag'][i] == ParticleTAGS.Real or pc['type'][i] == ParticleTAGS.Boundary or pc['type'][i] == ParticleTAGS.BoundarySecond:
            colors.append(pc["density"][i])

    fig, ax = plt.subplots()
    p = PatchCollection(l, alpha=0.4)
    p.set_array(np.array(colors))
    p.set_clim([0, 4])

    ax.set_xlim(-.1,1.1)
    ax.set_ylim(-.1,1.1)
    ax.add_collection(p)
    ax.set_aspect('equal')

#    current_axis = plt.gca()
#    for node in load_balance.global_tree.dump_data():
#        x = node[0]/2.0**load_balance.order
#        y = node[1]/2.0**load_balance.order
#        w = node[2]/2.0**load_balance.order
#        current_axis.add_patch(Rectangle((x-.5*w, y-.5*w), w, w, fill=None))

    tag = pc['type']
#    ghost = tag == 0
#    plt.scatter(pc['position-x'][ghost], pc['position-y'][ghost], marker=".", color='lightsteelblue')
    exterior = tag == 2
    plt.scatter(pc['position-x'][exterior], pc['position-y'][exterior], marker=".", color='red')
    exterior = tag == 8
    plt.scatter(pc['position-x'][exterior], pc['position-y'][exterior], marker=".", color='cyan')


    plt.savefig("plot_init_proc_%d.pdf" % rank, format='pdf')
    plt.clf()
