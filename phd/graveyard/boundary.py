import migrate
import numpy as np

from utils.particle_tags import ParticleTAGS
from utils.carray import IntArray, LongArray
from mesh.voronoi_mesh import VoronoiMesh2D
from utils.exchange_particles import exchange_particles
from containers.containers import CarrayContainer, ParticleContainer

def find_boundary_particles(indices, ghost_indices, total_ghost_indices,
        neighbors_graph, neighbors_graph_size, cumsum_neighbors, clean=True):
    """
    Find border particles, two layers, and return their indicies.
    """

    if clean:
        indices.reset()

    # grab all neighbors of ghost particles, this includes border cells
    border = set()
    for i in ghost_indices:
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]
        border.update(neighbors_graph[start:end])

    # grab neighbors again, this includes another layer of border cells 
    border_tmp = set(border)
    for i in border_tmp:
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]
        border.update(neighbors_graph[start:end])

    border_tmp = set(border)
    for i in border_tmp:
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]
        border.update(neighbors_graph[start:end])

    border_tmp = set(border)
    for i in border_tmp:
        start = cumsum_neighbors[i] - neighbors_graph_size[i]
        end   = cumsum_neighbors[i]
        border.update(neighbors_graph[start:end])

    # remove ghost particles leaving border cells that will create new
    # ghost particles
    border = border.difference(total_ghost_indices)

    for i in border:
        indices.append(i)

    return len(border)


def reflect_corner(parray, indices, sub_indices, x_qm, y_qm, ghost_indices,
        neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank):

    find_boundary_particles(indices, sub_indices.get_npy_array(), ghost_indices,
            neighbors_graph, neighbors_graph_size, cumsum_neighbors)
    copy = parray.extract_items(indices.get_npy_array())
    num_new = copy.get_number_of_items()

    copy['position-x'][:] -= 2*(copy['position-x'] - x_qm)
    copy['position-y'][:] -= 2*(copy['position-y'] - y_qm)
    copy['velocity-x'][:] *= -1.0
    copy['velocity-y'][:] *= -1.0

    copy['tag'][:] = ParticleTAGS.Ghost
    copy['process'][:] = rank
    parray.append_container(copy)

    return num_new


def reflect_axis(parray, indices, sub_indices, axis, qminmax, ghost_indices,
        neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank):

    find_boundary_particles(indices, sub_indices.get_npy_array(), ghost_indices,
            neighbors_graph, neighbors_graph_size, cumsum_neighbors)

    copy = parray.extract_items(indices.get_npy_array())
    num_new = copy.get_number_of_items()

    copy['position-%s' % axis][:] -= 2*(copy['position-%s' % axis] - qminmax)
    copy['velocity-%s' % axis][:] *= -1.0

    copy['tag'][:] = ParticleTAGS.Ghost
    copy['process'][:] = rank
    parray.append_container(copy)

    return num_new

def reflect_axis2(parray, ghost_pc, indices, sub_indices, axis, qminmax, ghost_indices,
        neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank):

    find_boundary_particles(indices, sub_indices, ghost_indices,
            neighbors_graph, neighbors_graph_size, cumsum_neighbors)

    copy = parray.extract_items(indices.get_npy_array())
    num_new = copy.get_number_of_items()

    copy['position-%s' % axis][:] -= 2*(copy['position-%s' % axis] - qminmax)
    copy['velocity-%s' % axis][:] *= -1.0

    copy['tag'][:] = ParticleTAGS.Ghost
    copy['process'][:] = rank
    ghost_pc.append_container(copy)

    return num_new

def export_reflect(particles, new_ghost, left, indices, send_particles, axis, qminmax, ghost_indices,
        neighbors_graph, neighbors_graph_size, cumsum_neighbors, load_balance, current_size, rank, size):

    ghost = particles.extract_items(left.get_npy_array())

    # shift particles and find which old ghost live on other process
    ghost['position-%s' % axis][:] += 2*(qminmax - ghost['position-%s' % axis])

    # update particle process to find which particles live on different process
    load_balance.update_particle_process(ghost, rank)

    # filter out local particles
    mask = ghost['process'] != rank
    gind = left.get_npy_array()[mask]
    proc = ghost['process'][mask]

    # put process in order
    ind = np.argsort(proc)
    gind = gind[ind]
    proc = proc[ind]

    # bin process
    proc_bin = np.bincount(proc, minlength=size)

    # collect the indices of particles to be export to each process
    cumsum_proc = proc_bin.cumsum()
    for p in range(size):
        if proc_bin[p] != 0:

            start = cumsum_proc[p] - proc_bin[p]
            end   = cumsum_proc[p]

            send_particles[p] += reflect_axis2(particles, new_ghost, indices, gind[start:end], axis, qminmax, ghost_indices,
                neighbors_graph, neighbors_graph_size, cumsum_neighbors, p)


def create_reflect_ghost(parray, boundary_indices, domain, exterior_ghost_indices,
        ghost_indices, neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank):

    indices = LongArray()

    # sides
    left   = boundary_indices["left"]
    right  = boundary_indices["right"]
    bottom = boundary_indices["bottom"]
    top    = boundary_indices["top"]

    # corners
    lt = boundary_indices["left-top"]
    lb = boundary_indices["left-bottom"]
    rt = boundary_indices["right-top"]
    rb = boundary_indices["right-bottom"]

    xmin = domain.xmin
    xmax = domain.xmax
    ymin = domain.ymin
    ymax = domain.ymax

    x = parray['position-x']
    y = parray['position-y']

    for i in exterior_ghost_indices:

        xi = x[i]; yi = y[i]

        # left boundary condition
        if xi < xmin:

            # left top corner
            if yi > ymax:
                lt.append(i)
            # left bottom corner
            elif yi < ymin:
                lb.append(i)
            # left
            else:
                left.append(i)

        # right boundary condition
        elif xi > xmax:

            # left top corner
            if yi > ymax:
                rt.append(i)
            # left bottom corner
            elif yi < ymin:
                rb.append(i)
            # right
            else:
                right.append(i)

        # bottom boundary condition
        elif yi < ymin:
            bottom.append(i)

        # top boundary condition
        elif yi > ymax:
            top.append(i)

    num_new_ghost = 0

    # left 
    if left.length > 0:
        num_new_ghost += reflect_axis(parray, indices, left, 'x', xmin, ghost_indices,
                neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank)

   # right
    if right.length > 0:
        num_new_ghost += reflect_axis(parray, indices, right, 'x', xmax, ghost_indices,
                neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank)

    # bottom
    if bottom.length > 0:
        num_new_ghost += reflect_axis(parray, indices, bottom, 'y', ymin, ghost_indices,
                neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank)

    # top
    if top.length > 0:
        num_new_ghost += reflect_axis(parray, indices, top, 'y', ymax, ghost_indices,
                neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank)

    # left top
    if lt.length > 0:
        num_new_ghost += reflect_corner(parray, indices, lt, xmin, ymax, ghost_indices,
                neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank)

    # left bottom
    if lb.length > 0:
        num_new_ghost += reflect_corner(parray, indices, lb, xmin, ymin, ghost_indices,
                neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank)

    # right top
    if rt.length > 0:
        num_new_ghost += reflect_corner(parray, indices, rt, xmax, ymax, ghost_indices,
                neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank)

    # right bottom
    if rb.length > 0:
        num_new_ghost += reflect_corner(parray, indices, rb, xmax, ymin, ghost_indices,
                neighbors_graph, neighbors_graph_size, cumsum_neighbors, rank)

    return num_new_ghost


class SingleCoreBoundary(object):

    def update_ghost_particles(self, particles, mesh, domain):

        # relabel all particles
        self.flag_exterior_particles(particles, domain.xmin, domain.xmax)

        boundary_indices = {
                "left"   : LongArray(),
                "right"  : LongArray(),
                "bottom" : LongArray(),
                "top"    : LongArray(),
                "left-top"     : LongArray(),
                "left-bottom"  : LongArray(),
                "right-top"    : LongArray(),
                "right-bottom" : LongArray()
                }

        # create new exterior boundary ghost particles
        cumsum_neighbors = mesh["number of neighbors"].cumsum()
        ghost_indices = np.where(particles["tag"] == ParticleTAGS.OldGhost)[0]
        create_reflect_ghost(particles, boundary_indices, domain, ghost_indices,
                ghost_indices, mesh['neighbors'], mesh['number of neighbors'],
                cumsum_neighbors, -1)

        # remove old ghost particles
        particles.remove_tagged_particles(ParticleTAGS.OldGhost)

        # place real particles in begginning of the array
        particles.align_particles()
        particles['type'][:] = ParticleTAGS.Undefined

    def flag_exterior_particles(self, pc, lo, hi):

        x = pc['position-x']
        y = pc['position-y']

        # find all particles in the interior domain
        indices = (((lo <= x) & (x <= hi)) & ((lo <= y) & (y <= hi)))
        pc['tag'][indices]  = ParticleTAGS.Real
        pc['tag'][~indices] = ParticleTAGS.OldGhost


class MultiCoreBoundary(object):

    def create_ghost_particles(self, particles, mesh, domain, load_balance, comm, iteration=6):
        """Create initial ghost particles that hug the boundary after
        load balance
        """
        rank = comm.Get_rank()
        size = comm.Get_size()

        # remove current (if any) ghost particles
        particles.remove_tagged_particles(ParticleTAGS.Ghost)
        current_size = particles.get_number_of_particles()

        # create initial ghost particles, particles is now larger
        # these particles are centered in neighboring boundary leaf
        # cells of the octree
        load_balance.create_boundary_particles(particles, rank)

        # reorder ghost in processors order: exterior have a process id of -1 so
        # their put before interior ghost particles
        ghost_proc = np.array(particles["process"][current_size:])
        ind = np.argsort(ghost_proc)
        ghost_proc = ghost_proc[ind]

        for field in particles.properties.keys():
            array = particles[field][current_size:]
            array[:] = array[ind]

        # allocate arrays for boundary indices
        indices = LongArray()
        corner_ghost = ParticleContainer()

        # sides
        boundary_indices = {
                "left"   : LongArray(),
                "right"  : LongArray(),
                "bottom" : LongArray(),
                "top"    : LongArray(),
                "left-top"     : LongArray(),
                "left-bottom"  : LongArray(),
                "right-top"    : LongArray(),
                "right-bottom" : LongArray()
                }

        send_particles = np.zeros(size, dtype=np.int32)
        recv_particles = np.zeros(size, dtype=np.int32)

        # create ghost interior and exterior particles by iteration, using
        # the mesh to extract the needed neighbors
        for i in range(iteration):

            # build the mesh
            mesh.tessellate()
            cumsum_neighbors = mesh["number of neighbors"].cumsum()

            #---------- create exterior ghost particles ----------#

            # create indices for ghost particles
            ghost_indices = np.arange(current_size, particles.get_number_of_particles())

            # label current ghost as old ghost
            particles['tag'][ghost_indices] = ParticleTAGS.OldGhost

            # select exterior ghost particles
            exterior_ghost = ghost_proc == -1
            exterior_ghost_indices = ghost_indices[exterior_ghost]

            if np.sum(exterior_ghost_indices) > 0:

                num_exterior_ghost = create_reflect_ghost(particles, boundary_indices,
                        domain, exterior_ghost_indices, ghost_indices,
                        mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, -1)

            #---------- create interior ghost particles ----------#
            interior_ghost_indices = ghost_indices[~exterior_ghost]
            interior_ghost_proc = ghost_proc[~exterior_ghost]

            # bin processors - they are in order
            interior_ghost_proc_bin = np.bincount(interior_ghost_proc, minlength=size)

            send_particles[:] = 0
            recv_particles[:] = 0
            indices.reset()

            # collect the indices of particles to be export to each process
            cumsum_proc = interior_ghost_proc_bin.cumsum()
            for proc in range(size):
                if interior_ghost_proc_bin[proc] != 0:

                    start = cumsum_proc[proc] - interior_ghost_proc_bin[proc]
                    end   = cumsum_proc[proc]

                    send_particles[proc] = find_boundary_particles(indices, interior_ghost_indices[start:end], ghost_indices,
                            mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, False)

            # extract data to send and remove the particles
            send_data = {}
            for prop in particles.properties.keys():
                send_data[prop] = np.ascontiguousarray(particles[prop][indices.get_npy_array()])
            send_data["tag"][:] = ParticleTAGS.Ghost

            # how many particles are being sent from each process
            comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)
            num_interior_ghost = np.sum(recv_particles)

            # resize arrays to give room for incoming particles
            sp = particles.get_number_of_particles()
            #particles.resize(current_size + num_exterior_ghost + num_interior_ghost)
            particles.extend(num_interior_ghost)

            exchange_particles(particles, send_data, send_particles, recv_particles,
                    sp, comm)

            #---------- create exterior corner ghost particles ----------#
            indices.reset()
            send_particles[:] = 0
            recv_particles[:] = 0

            # clear out corner ghost
            corner_ghost.resize(0)

            if boundary_indices['left'].length > 0:
                export_reflect(particles, corner_ghost, boundary_indices["left"], indices, send_particles, 'x', domain.xmin, ghost_indices,
                        mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, load_balance, current_size, rank, size)

            if boundary_indices['right'].length > 0:
                export_reflect(particles, corner_ghost, boundary_indices["right"], indices, send_particles, 'x', domain.xmax, ghost_indices,
                        mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, load_balance, current_size, rank, size)

            if boundary_indices['bottom'].length > 0:
                export_reflect(particles, corner_ghost, boundary_indices["bottom"], indices, send_particles, 'y', domain.ymin, ghost_indices,
                        mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, load_balance, current_size, rank, size)

            if boundary_indices['top'].length > 0:
                export_reflect(particles, corner_ghost, boundary_indices["top"], indices, send_particles, 'y', domain.ymax, ghost_indices,
                        mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, load_balance, current_size, rank, size)

            #print rank, current_size, particles.get_number_of_particles(), current_size + num_exterior_ghost + num_interior_ghost

            sp = particles.get_number_of_particles()
            comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)
            particles.extend(np.sum(recv_particles))

            # to export corners have to be reorderd by process
            if corner_ghost.get_number_of_particles() > 0:

                ind = np.argsort(corner_ghost['process'])
                for field in corner_ghost.properties.keys():
                    array = corner_ghost[field]
                    array[:] = array[ind]

                corner_ghost["process"][:] = -1

            # exchange patch corners
            exchange_particles(particles, corner_ghost, send_particles, recv_particles,
                    sp, comm)

            for bd in boundary_indices:
                boundary_indices[bd].reset()

            particles.remove_tagged_particles(ParticleTAGS.OldGhost)

            # put particles in process order for next loop
            ind = np.argsort(particles["process"][current_size:])
            for field in particles.properties.keys():
                array = particles[field][current_size:]
                array[:] = array[ind]

            ghost_proc = np.array(particles["process"][current_size:])

        print 'rank:', rank, 'fraction of real to ghost:', (particles.get_number_of_particles()-current_size)*1.0/particles.get_number_of_particles()


    def update_ghost_particles(self, particles, mesh, domain, load_balance, comm):

        rank = comm.Get_rank()
        size = comm.Get_size()

        # allocate arrays for boundary indices
        indices = LongArray()
        corner_ghost = ParticleContainer()

        send_particles = np.zeros(size, dtype=np.int32)
        recv_particles = np.zeros(size, dtype=np.int32)

        # we are puting new ghost at the end of the array
        current_size = particles.get_number_of_particles()

        boundary_indices = {
                "left"   : LongArray(),
                "right"  : LongArray(),
                "bottom" : LongArray(),
                "top"    : LongArray(),
                "left-top"     : LongArray(),
                "left-bottom"  : LongArray(),
                "right-top"    : LongArray(),
                "right-bottom" : LongArray()
                }

        # relabel all particles
        particles["tag"][:]  = ParticleTAGS.Undefined
        particles["type"][:] = ParticleTAGS.Undefined

        # flag particles that have left the domain and particles
        # that remained
        load_balance.flag_migrate_particles(particles, rank)

        # find particles that have left the domain
        export_indices = np.where(particles["type"] == ParticleTAGS.ExportInterior)[0]

        if export_indices.size > 0:

            # extract export particles 
            export_particles = particles.extract_items(export_indices)

            # put particles in process order
            ind = np.argsort(export_particles["process"])
            for field in export_particles.properties.keys():
                array = export_particles[field]
                array[:] = array[ind]

            export_particles["tag"][:]  = ParticleTAGS.Real
            export_particles["type"][:] = ParticleTAGS.Undefined

        else:
            export_particles = ParticleContainer()

        # bin particle process
        send_particles[:] = np.bincount(export_particles["process"], minlength=size)

        # how many particles are being sent from each process
        comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)

        # create container for incoming particles 
        import_particles = ParticleContainer(np.sum(recv_particles))

        exchange_particles(import_particles, export_particles, send_particles, recv_particles,
                0, comm)

        # copy import particle data to ghost place holders and turn to real particles
        migrate.transfer_migrate_particles(particles, import_particles)

        # flag export particles back to interior ghost particles
        particles["type"][export_indices] = ParticleTAGS.Interior

        ghost_indices = np.where(particles["tag"] == ParticleTAGS.OldGhost)[0]

        # find indices of interior/exterior ghost particles 
        cumsum_neighbors = mesh["number of neighbors"].cumsum()
        exterior_ghost_indices = np.where(particles["type"] == ParticleTAGS.Exterior)[0]
        interior_ghost_indices = np.where(particles["type"] == ParticleTAGS.Interior)[0]

        #---------- create exterior ghost particles ----------#
        num_exterior_ghost = 0
        if exterior_ghost_indices.size > 0:

            num_exterior_ghost = create_reflect_ghost(particles, boundary_indices,
                    domain, exterior_ghost_indices, ghost_indices,
                    mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, -1)

        #---------- create interior ghost particles ----------#
        send_particles[:] = 0
        recv_particles[:] = 0
        interior_ghost_proc = particles["process"][interior_ghost_indices]

        # arrange particles in process order
        ind = interior_ghost_proc.argsort()
        interior_ghost_proc = interior_ghost_proc[ind]
        interior_ghost_indices = interior_ghost_indices[ind]

        # bin processors
        interior_ghost_proc_bin = np.bincount(interior_ghost_proc, minlength=size)

        cumsum_neighbors = mesh["number of neighbors"].cumsum()

        # collect the indices of particles to be export to each process
        cumsum_proc = interior_ghost_proc_bin.cumsum()
        for proc in range(size):
            if interior_ghost_proc_bin[proc] != 0:

                start = cumsum_proc[proc] - interior_ghost_proc_bin[proc]
                end   = cumsum_proc[proc]

                send_particles[proc] = find_boundary_particles(indices, interior_ghost_indices[start:end], ghost_indices,
                        mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, False)

        # extract data to send and remove the particles
        send_data = {}
        for prop in particles.properties.keys():
            send_data[prop] = np.ascontiguousarray(particles[prop][indices.get_npy_array()])
        send_data["tag"][:] = ParticleTAGS.Ghost

        # how many particles are being sent from each process
        comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)
        num_interior_ghost = np.sum(recv_particles)

        # resize arrays to give room for incoming particles
        sp = particles.get_number_of_particles()
        particles.extend(num_interior_ghost)

        exchange_particles(particles, send_data, send_particles, recv_particles,
                sp, comm)

        #---------- create exterior corner ghost particles ----------#
        indices.reset()
        send_particles[:] = 0
        recv_particles[:] = 0

        if boundary_indices['left'].length > 0:
            export_reflect(particles, corner_ghost, boundary_indices["left"], indices, send_particles, 'x', domain.xmin, ghost_indices,
                    mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, load_balance, current_size, rank, size)

        if boundary_indices['right'].length > 0:
            export_reflect(particles, corner_ghost, boundary_indices["right"], indices, send_particles, 'x', domain.xmax, ghost_indices,
                    mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, load_balance, current_size, rank, size)

        if boundary_indices['bottom'].length > 0:
            export_reflect(particles, corner_ghost, boundary_indices["bottom"], indices, send_particles, 'y', domain.ymin, ghost_indices,
                    mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, load_balance, current_size, rank, size)

        if boundary_indices['top'].length > 0:
            export_reflect(particles, corner_ghost, boundary_indices["top"], indices, send_particles, 'y', domain.ymax, ghost_indices,
                    mesh['neighbors'], mesh['number of neighbors'], cumsum_neighbors, load_balance, current_size, rank, size)

        comm.Alltoall(sendbuf=send_particles, recvbuf=recv_particles)
        sp = particles.get_number_of_particles()
        particles.extend(np.sum(recv_particles))

        # to export corners have to be reorderd by process
        if corner_ghost.get_number_of_particles() > 0:

            ind = np.argsort(corner_ghost['process'])
            for field in corner_ghost.properties.keys():
                array = corner_ghost[field]
                array[:] = array[ind]

            corner_ghost["process"][:] = -1

        # exchange patch corners
        exchange_particles(particles, corner_ghost, send_particles, recv_particles,
                sp, comm)

        # finally remove old ghost particles from previous time step
        # and also put real particles in front and ghost in the back
        particles.remove_tagged_particles(ParticleTAGS.OldGhost)
        particles.align_particles()

        particles['type'][:] = ParticleTAGS.Undefined
