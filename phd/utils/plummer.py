import phd
import numpy as np
import matplotlib.pyplot as plt

def plummer(r, M=1.):
    return 3.*M/4./np.pi*(1.+r**2)**(-5./2.)

def make_plummer(particles, M=1., R=1.):

    N = particles.get_carray_size()

    E = 3./64.*np.pi*M*M/R
    np.random.seed(0)
    i = 0; count = 0
    while i < N:

        count += 1

        x1 = np.random.uniform()
        x2 = np.random.uniform()
        x3 = 2*np.pi*np.random.uniform()

        r = (x1**(-2./3.)-1.)**(-1./2.)

        if r > 30:
            continue

        z = (1.-2.*x2)*r
        x = np.sqrt(r*r - z*z)*np.cos(x3)
        y = np.sqrt(r*r - z*z)*np.sin(x3)

        x5 = 0.1
        q  = 0.0
        while x5 > q*q*(1.-q*q)**3.5:

            x5 = 0.1*np.random.uniform()
            q = np.random.uniform()

        ve = np.sqrt(2)*(1. + r*r)**(-1./4.)
        v = q*ve

        x6 = np.random.uniform()
        x7 = 2*np.pi*np.random.uniform()

        vz = (1.-2.*x6)*v
        vx = np.sqrt(v*v - vz*vz)*np.cos(x7)
        vy = np.sqrt(v*v - vz*vz)*np.sin(x7)

        x *= 3.*np.pi/64.*M*M/E
        y *= 3.*np.pi/64.*M*M/E
        z *= 3.*np.pi/64.*M*M/E

        vx *= np.sqrt(E*64./3./np.pi/M)
        vy *= np.sqrt(E*64./3./np.pi/M)
        vz *= np.sqrt(E*64./3./np.pi/M)

        particles["position-x"][i] = x
        particles["position-y"][i] = y
        particles["position-z"][i] = z

        particles["velocity-x"][i] = vx
        particles["velocity-y"][i] = vy
        particles["velocity-z"][i] = vz

        i += 1

    particles["mass"][:] = M/count

if __name__ == "__main__":

    num_part = 10000
    particles = phd.CarrayContainer(n=num_part)
    particles.register_carray(num_part, "position-x", "double")
    particles.register_carray(num_part, "position-y", "double")
    particles.register_carray(num_part, "position-z", "double")
    particles.register_carray(num_part, "velocity-x", "double")
    particles.register_carray(num_part, "velocity-y", "double")
    particles.register_carray(num_part, "velocity-z", "double")
    particles.register_carray(num_part, "mass", "double")

    make_plummer(particles, 1000.)

    totmass = 0.
    com = [0., 0., 0.]
    for i in range(particles.get_carray_size()):
        for j, ax in enumerate("xyz"):
            com[j] += particles["mass"][i]*particles["position-"+ax][i]
        totmass += particles["mass"][i]
    for j in range(3):
        com[j] /= totmass

    radius = np.zeros(particles.get_carray_size())
    for i in range(particles.get_carray_size()):
        radius[i] = np.sqrt(
                (particles["position-x"][i]-com[0])**2 +\
                (particles["position-y"][i]-com[1])**2 +\
                (particles["position-z"][i]-com[2])**2)

    rmin, rmax = min(radius), max(radius)
    dr = 0.01*(rmax - rmin)
    r = np.arange(rmin, rmax + dr, dr)

    dens = np.zeros(r.size)
    for i in range(particles.get_carray_size()):
        rad = np.sqrt(
                (particles["position-x"][i]-com[0])**2 +\
                (particles["position-y"][i]-com[1])**2 +\
                (particles["position-z"][i]-com[2])**2)

        index = 0
        while index+1 < len(r) and rad > r[index+1]:
            index += 1
        dens[index] += particles["mass"][i]

    for i in range(len(r)):
        dens[i] /= (4*np.pi/3.*(3*r[i]*dr**2 + 3.*r[i]**2*dr + dr**3))

    r_pl = np.logspace(-2, 2)
    plt.title("Plummer Potential")
    plt.loglog(r_pl, plummer(r_pl, 1000.))
    plt.loglog(r, dens, 'g^')
    plt.xlim(1.0E-2, 100)
    plt.ylim(1.0E-5, 1.0E3)
    plt.xlabel("r")
    plt.ylabel(r"$\rho$")
    plt.savefig("plummer_model.png")
    plt.show()
