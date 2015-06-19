class boundary(object):

    def create_ghost_particles(self, particles, leaf_proc, global_tree,
            boundaries, corner, box_length, comm, order):
        """Create initial ghost particles that hug the boundary
        """
        rank = comm.Get_rank()
        size = comm.Get_size()

        # create temp ghost particles
        global_tree.create_boundary_particles(particles, rank, leaf_proc)

        # reorder in processors order: exterior are put before interior ghost particles
        proc_id = ghost_particles[2,:].astype(np.int32)
        ind = np.argsort(proc_id)
        ghost_particles = ghost_particles[:2,ind]
        proc_id = proc_id[ind]

        # create new position array for temporary mesh
        particles.discard_ghost_particles()
        current_size = particles.num_real_particles
        new_size = current_size + ghost_particles.shape[1]
        particles.resize(new_size)

        # add this to the particle container instead
        particles['position-x'][current_size:] = ghost_particles[0,:]/2.0**order
        particles['position-y'][current_size:] = ghost_particles[1,:]/2.0**order

        boundaries = [[self.corner[0], self.corner[0]+self.box_length],
                [self.corner[1], self.corner[1]+self.box_length]]




        mesh = VoronoiMesh2D()
        for i in range(5):

            # build the mesh
            p = np.array([particles['position-x'], particles['position-y']])
            #graphs = mesh.tessellate(p)
            mesh.tessellate(p)

            ghost_indices = np.arange(current_size, new_size)

            # select exterior ghost particles
            exterior_ghost = proc_id == -1
            num_exterior_ghost = np.sum(exterior_ghost)
            exterior_ghost_indices = ghost_indices[exterior_ghost]

            new_ghost = np.empty((2,0), dtype=np.float64)
            old_ghost = p[:,current_size:current_size+num_exterior_ghost]
            num_exterior_ghost = 0 # count new exterior

#            for k, bound in enumerate(boundaries):
#
#                do_min = True
#                for qm in bound:
#
#                    if do_min == True:
#                        # lower boundary 
#                        i = np.where(old_ghost[k,:] < qm)[0]
#                        do_min = False
#                    else:
#                        # upper boundary
#                        i = np.where(qm < old_ghost[k,:])[0]
#
#                    # find bordering real particles
#                    border = find_boundary_particles(mesh['neighbors'], mesh['number of neighbors'],
#                            exterior_ghost_indices[i], ghost_indices)
#
#                    if border.size != 0:
#
#                        # allocate space for new ghost particles
#                        tmp = np.empty((2, len(border)), dtype=np.float64)
#
#                        # reflect particles across boundary
#                        tmp[:,:] = p[:,border]
#                        tmp[k,:] = 2*qm - p[k,border]
#
#                        # add the new ghost particles
#                        new_ghost = np.concatenate((new_ghost, tmp), axis=1)
#                        num_exterior_ghost += border.size
#
#            new_size = current_size + num_exterior_ghost
#            particles.resize(new_size)
#
#            # add in the new exterior ghost particles
#            particles['position-x'][current_size:current_size+num_exterior_ghost] = new_ghost[0,:]
#            particles['position-y'][current_size:current_size+num_exterior_ghost] = new_ghost[1,:]


            # do the interior particles now
            interior_ghost_indices = ghost_indices[~exterior_ghost]
            interior_proc_id = proc_id[~exterior_ghost]

            # bin processors
            interior_proc_bin = np.bincount(interior_proc_id, minlength=size)
            send_particles = np.zeros(size, dtype=np.int32)

            # collect the indices for particles to export for each process
            ghost_list = []
            cumsum_proc = interior_proc_bin.cumsum()
            for proc in xrange(size):
                if interior_proc_bin[proc] != 0:

                    start = cumsum_proc[proc] - interior_proc_bin[proc]
                    end   = cumsum_proc[proc]

                    border = find_boundary_particles(mesh['neighbors'], mesh['number of neighbors'],
                            interior_ghost_indices[start:end], ghost_indices)

                    ghost_list.append(border)
                    send_particles[proc] = border.size

            # flatten out the indices
            new_ghost = np.array(list(itertools.chain.from_iterable(ghost_list)), dtype=np.int32)

            # extract data to send and remove the particles
            send_data = {}
            for prop in particles.properties.keys():
                send_data[prop] = np.ascontiguousarray(particles[prop][new_ghost])

            # discard current ghost particles: this does not release memory used
            particles.discard_ghost_particles()

            # how many particles are being sent from each process
            recv_particles = np.empty(size, dtype=np.int32)
            comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)

            # resize arrays to give room for incoming particles
            new_size = current_size + num_exterior_ghost + np.sum(recv_particles)
            particles.resize(new_size)

            #print "rank: %d iteration: %d num: %d" % (rank, i, np.sum(recv_particles))

            self.exchange_particles(particles, send_data, send_particles, recv_particles,
                    current_size + num_exterior_ghost)

            # temp hack 
            proc_id = np.concatenate((-1*np.ones(num_exterior_ghost), np.repeat(np.arange(size),
                recv_particles))).astype(np.int32)

        return init_border

def create_reflect_ghost(ParticleArray particles, Mesh mesh, domain):

    cdef LongArray indices = LongArray()

    cdef LongArray left = LongArray()
    cdef LongArray right = LongArray()
    cdef LongArray bottom = LongArray()
    cdef LongArray top = LongArray()

    cdef double xi, yi
    cdef double xmin, xmax, ymin, ymax

    xmin = domain.xmin
    xmax = domain.xmax
    ymin = domain.ymin
    ymay = domain.ymax

    cdef int np = particles.tot_number_particles()
    for i in xrange(np):
        # only use boundary particles
        if type[i] >= 4:
            xi = x.data[i]; yi = y.data[i]
            ghost.append(i)

            # left boundary condition
            if xi < xmin:
                left.append(i)

            # right boundary condition
            if xi > xmax:
                right.append(i)

            # bottom boundary condition
            if yi < ymin:
                bottom.append(i)

            # top boundary condition
            if yi > ymax:
                top.append(i)

    # find corresponding real particles to make ghost
    find_boundary(indices, left)
    copy = particles.extract_particles(indices)
    copy["position-x"] -= 2*(copy["position-x"] - xmin)
    particles.append(copy)

    find_boundary(indices, right)
    copy = particles.extract_particles(indices)
    copy["position-x"] += 2*(xmax - copy["position-x"])
    particles.append(copy)

    find_boundary(indices, bottom)
    copy = particles.extract_particles(indices)
    copy["position-y"] -= 2*(copy["position-y"]-ymin)
    particles.append(copy)

    find_boundary(indices, top)
    copy = particles.extract_particles(indices)
    copy["position-y"] += 2*(ymax - copy["position-y"])
    particles.append(copy)

#cdef find_boundary_particles(neighbor_graph, neighbors_graph_size, ghost_indices, total_ghost_indices):
cdef find_boundary_particles(LongArray indices, LongArray ghost_indices, LongArray total_ghost_indices,
        np.int64_t[:] cumsum_neighbors, np.int64_t[:] neighbors_graph_size)
    """
    Find border particles, two layers, and return their indicies.
    """
    indicies.reset()

    # grab all neighbors of ghost particles, this includes border cells
    cdef int i, start, end
    cdef set border = set()
    for i in ghost_indices:
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]
        border.update(neighbor_graph[start:end])

    # grab neighbors again, this includes another layer of border cells 
    cdef list border_tmp = list(border)
    for i in border_tmp:
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]
        border.update(neighbor_graph[start:end])

    # remove ghost particles leaving border cells that will create new ghost particles
    border = border.difference(total_ghost_indices)

    for i in border:
        indicies.append(i)
