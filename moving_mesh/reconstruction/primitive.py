import numpy as np

def primitive_variables(data, volume, gamma):
  
    # conserative vector is mass, momentum, total energy
    prim = np.zeros(data.shape)

    # mass
    mass = data[0,:]


    # density
    prim[0,:] = data[0,:]/volume
    
    # velocity
    prim[1:3,:] = data[1:3,:]/mass

    # pressure
    prim[3,:] = (data[3,:]/volume - 0.5*data[0,:]*(prim[1,:]**2 + prim[2,:]**2))*(gamma-1.0)

    return prim

def conservative_variables(data, volume):

    data[0,:] = data[0,:]*volume
    data[3,:] = data[3,:]*volume
