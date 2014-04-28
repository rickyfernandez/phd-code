import numpy as np

def update(data, fluxes, dt, face_info, particle_index):
    #d2 = data.copy()

    ghost_map = particle_index["ghost_map"]
    area = face_info[1,:]

    #NOT_GHOST4 = ~np.in1d(face_info[4,:], particle_index["ghost_map"].keys())
    #NOT_GHOST5 = ~np.in1d(face_info[5,:], particle_index["ghost_map"].keys())
    k = 0
    for i, j in zip(face_info[4,:], face_info[5,:]):

        # do not update ghost particle cells
        #if NOT_GHOST4[k]:
        data[:,i] -= dt*area[k]*fluxes[:,k]

        # do not update ghost particle cells
        #if NOT_GHOST5[k]:
        if not ghost_map.has_key(j):
            data[:,j] += dt*area[k]*fluxes[:,k]

        k += 1
