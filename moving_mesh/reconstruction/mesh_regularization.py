import numpy as np

def mesh_regularization(prim, particles, gamma, vol_center_mass, particles_index):

    eta = 0.25

    indices = particles_index["real"]

    pressure = prim[3, indices]
    rho      = prim[0, indices]

    c = np.sqrt(gamma*pressure/rho)
    
    # generate distance for center mass to particle position
    r = np.transpose(particles[indices])
    s = vol_center_mass[1:3,:]

    d = s - r
    d = np.sqrt(np.sum(d**2,axis=0))

    R = np.sqrt(vol_center_mass[0,:]/np.pi)

    #w = np.copy(prim[1:3, indices])
    w = np.zeros(s.shape)


    i = (0.9 <= d/(eta*R)) & (d/(eta*R) < 1.1)
    if i.any():
        w[:,i] += c[i]*(s[:,i] - r[:,i])*(d[i] - 0.9*eta*R[i])/(d[i]*0.2*eta*R[i])

    j = 1.1 <= d/(eta*R)
    if j.any():
        w[:,j] += c[j]*(s[:,j] - r[:,j])/d[j]

    return w

