import numpy as np

def time_step(prim, vol, gamma, CFL):
  
    # calculate sound speed
    c = np.sqrt(gamma*prim[3,:]/prim[0,:])
    
    # calculate approx radius of each voronoi cell
    R = np.sqrt(vol/np.pi)

    return CFL*np.min(R/c)
