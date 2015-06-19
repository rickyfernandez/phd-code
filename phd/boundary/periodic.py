import numpy as np
from boundary_base import BoundaryBase


class Periodic2d(BoundaryBase):
    """
    2d refect boundary class
    """
    def __init__(self, xl, xr, yl, yr):

        self.dim = 2
        self.boundaries = [
                [xl, xr],   # x dim
                [yl, yr]    # y dim
                ]

    def update_boundaries(self, particles, particles_index, neighbor_graph, neighbor_graph_size):
        """
        create ghost particles from real particles using periodic boundary conditins.
        """
        real_indices = particles_index["real"]

        # real indices of previous time step 
        pos = particles[:,real_indices]

        # boundary positions
        xl, xy = self.boundary[0]
        yl, yy = self.boundary[1]

        # find real particles that have not left the domain
        i = np.where(((xl < pos[0,:]) & (pos[0,:] < xr)) & ((yl < pos[1,:]) & (pos[1,:] < yr)))[0]
        new_indices = real_indices[i]

        # find real particles that have left the boundary
        out_indices = np.setdiff1d(real_indices, new_indices)

        # put back the particles that have left the boundary
        for k, bound in enumerate(self.boundaries):

            do_min = True

            ql, qr = bound
            box_size = qr - ql

            for qm in bound:

                if do_min == True:

                    # lower boundary 
                    i = np.where(pos[k,out_indices] < qm)[0]

                    # put particles back in the domain 
                    particles[k,out_indices[i]] += box_size

                    do_min = False

                else:

                    # upper boundary
                    i = np.where(qm < pos[k,out_indices])[0]

                    # put particles back in the domain 
                    particles[k,out_indices[i]] -= box_size


        # now create new ghost particles
        ghost_indices = particles_index["ghost"]

        # position of old ghost 
        old_ghost = particles[:,ghost_indices]

        # arrays for position of new ghost particles
        new_ghost = np.empty((self.dim,0), dtype="float64")

        # hold all the indices of real particles used to create ghost particles
        ghost_mapping_indices = np.empty(0, dtype="int32")

        # grab all neighbors of ghost particles, this includes border cells and 
        # neighbors of border cells, then remove ghost particles leaving two layers
        for k, bound in enumerate(self.boundaries):

            do_min = True

            ql, qr = bound
            box_size = qr - ql

            for qm in bound:

                if do_min == True:
                    # lower boundary 
                    i = np.where(old_ghost[k,:] < qm)[0]
                    do_min = False
                else:
                    # upper boundary
                    i = np.where(qm < old_ghost[k,:])[0]
                    box_size *= -1

                # find bordering real particles
                border = self.find_boundary_particles(neighbor_graph, neighbor_graph_size, ghost_indices[i], ghost_indices)

                # allocate space for new ghost particles
                tmp = np.empty((self.dim, len(border)), dtype="float64")

                # move particles across boundary
                tmp[:,:]  = particles[:,border]
                tmp[k,:] += box_size

                # add the new ghost particles
                new_ghost = np.concatenate((new_ghost, tmp), axis=1)

                # add real particles indices correspoinding to ghost particles 
                ghost_mapping_indices = np.append(ghost_mapping_indices, border)


        # arrays for position of new ghost particles
        new_real = np.empty((self.dim,0), dtype="float64")

        # hold all the indices of real particles used to create ghost particles
        ghost_mapping_indices = np.empty(0, dtype="int32")

        # grab all neighbors of ghost particles, this includes border cells and 
        # neighbors of border cells, then remove ghost particles leaving two layers

        # create the new list of particles
        real_indices = particles_index["real"]

        # first the real particles
        new_particles = np.copy(particles[:,real_indices])

        # now the new ghost particles
        new_particles = np.concatenate((new_particles, new_ghost), axis=1)

        # update particle information, generate new ghost map
        ghost_map = {}
        for i,j in enumerate(np.arange(real_indices.size, new_particles.shape[1])):
            ghost_map[j] = ghost_mapping_indices[i]

        particles_index["ghost_map"] = ghost_map
        particles_index["ghost"] = np.arange(real_indices.size, new_particles.shape[1])

        return new_particles
