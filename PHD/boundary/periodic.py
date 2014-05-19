import numpy as np
from find_boundary_particles import find_boundary_particles
#
#
#
#  # CODE DOES NOT WORK RIGHT NOW!!!!
#
#

def periodic(particles, particles_index, old_data, neighbor_graph, boundary):
    """
    update the particles with periodic boundaries 
    """
    #---------------------------------------------------------------------------
    # find new current particles in the domain, ghost + old interior

    x = particles[:,0]; y = particles[:,1]

    # grab domain values
    left   = boundary["left"];   right = boundary["right"]
    bottom = boundary["bottom"]; top   = boundary["top"]

    # find all new interior particles
    i = np.where((left < x) & (x < right))[0]
    j = np.where((bottom < y) & (y < top))[0]
    k = np.intersect1d(i, j)

    # copy all the interior particles
    x_new_particles = np.copy(x[k])
    y_new_particles = np.copy(y[k])

    # find ghost particles that have become real particles
    first, last = particles_index["interior"]
    new_real_i  = np.setdiff1d(k, np.arange(first, last+1))

    #---------------------------------------------------------------------------
    # find interior particles that have become ghost particles 

    first, last = particles_index["interior"]
    indicies    = np.arange(first, last+1)
    
    x = particles[first:last+1,0]; y = particles[first:last+1,1]

    # new ghost particles
    new_ghost_i = np.empty(0, dtype=np.int32)

    # particles that left the right boundary are now ghost particles
    i = np.where(x > right)[0]
    if i.size: new_ghost_i = np.append(new_ghost_i, indicies[i])

    # particles that left the left boundary are now ghost particles
    i = np.where(x < left)[0]
    if i.size: new_ghost_i = np.append(new_ghost_i, indicies[i])

    # particles that left the top boundary are now ghost particles
    i = np.where(y > top)[0]
    if i.size: new_ghost_i = np.append(new_ghost_i, indicies[i])

    # particles that left the bottom boundary are now ghost particles
    i = np.where(y < bottom)[0]
    if i.size: new_ghost_i = np.append(new_ghost_i, indicies[i])

    #---------------------------------------------------------------------------
    # find border particles and generate corresponding ghost particles

    first, last    = particles_index["ghost"]
    ghost_indicies = np.arange(first, last+1)

    # remove ghost particles that have become real particles
    ghost_indicies = np.setdiff1d(ghost_indicies, new_real_i)

    # add interior particles that are new ghost particles
    ghost_indicies = np.append(ghost_indicies, new_ghost_i)

    xg = particles[ghost_indicies,0]; yg = particles[ghost_indicies,1]
    x = particles[:,0];               y = particles[:,1]

    x_ghost = np.empty(0)
    y_ghost = np.empty(0)

    # grab all neighbors of ghost particles, this includes border cells and 
    # neighbors of border cells, then remove ghost particles leaving two layers

    # left boundary
    i = np.where(right < xg)[0]
    border = find_boundary_particles(neighbor_graph, ghost_indicies[i], ghost_indicies)

    # reflect particles to the left boundary
    x_ghost = np.append(x_ghost, (left + right) - (x[border] - 2*(x[border]-right)))
    y_ghost = np.append(y_ghost, y[border])

    # add the old indicies
    k = np.append(k, border)


    # right boundary 
    i = np.where(xg < left)[0]
    border = find_boundary_particles(neighbor_graph, ghost_indicies[i], ghost_indicies)

    # reflect particles to the right boundary
    x_ghost = np.append(x_ghost, (left + right) - (x[border] + 2*(left-x[border])))
    y_ghost = np.append(y_ghost, y[border])

    # add the old indicies
    k = np.append(k, border)


    # bottom boundary 
    i = np.where(yg > top)[0]
    border = find_boundary_particles(neighbor_graph, ghost_indicies[i], ghost_indicies)

    # reflect particles to the bottom boundary
    x_ghost = np.append(x_ghost, x[border])
    y_ghost = np.append(y_ghost, (bottom + top) - (y[border] + 2*(top-y[border])))

    # add the old indicies
    k = np.append(k, border)


    # top boundary 
    i = np.where(yg < bottom)[0]
    border = find_boundary_particles(neighbor_graph, ghost_indicies[i], ghost_indicies)

    # reflect particles to the left boundary
    x_ghost = np.append(x_ghost, x[border])
    y_ghost = np.append(y_ghost, (bottom + top) - (y[border] - 2*(y[border]-bottom)))

    # add the old indicies
    k = np.append(k, border)

    #---------------------------------------------------------------------------
    # create the new list of particles

    # first the interior particles
    particles_index["interior"] = (0, x_new_particles.size - 1)
    first, last = particles_index["interior"]

    x_new_particles = np.append(x_new_particles, x_ghost)
    y_new_particles = np.append(y_new_particles, y_ghost)

    particles_index["ghost"] = (last+1, x_new_particles.size - 1)
    particles_index["total"] = (0,      x_new_particles.size - 1)

    return  np.array(zip(x_new_particles, y_new_particles)), old_data[k]
