import numpy as np
cimport numpy as np
cimport libc.stdlib as stdlib

from mpi4py import MPI
from libc.math cimport sqrt

from ..utils.particle_tags import ParticleTAGS
from ..utils.carray cimport DoubleArray, IntArray, LongArray, LongLongArray
from ..load_balance.tree cimport TreeMemoryPool as Pool


cdef class Interaction:

    def __init__(self, DomainLimits domain):
        self.dim = domain.dim

    cdef void interact(self, Node* node):
        msg = "InteractionBase::interact called!"
        raise NotImplementedError(msg)

    cdef void initialize_particles(self, CarrayContainer pc):
        msg = "InteractionBase::initialize_particles called!"
        raise NotImplementedError(msg)

    cdef int process_particle(self):
        msg = "InteractionBase::process_particle called!"
        raise NotImplementedError(msg)

    cpdef void set_splitter(self, Splitter splitter):
        self.splitter = splitter
        self.splitter.dim = self.dim

cdef class GravityAcceleration(Interaction):

    def __init__(self, DomainLimits domain):
        self.dim = domain.dim

    cdef void interact(self, Node* node):
        cdef int i, inside
        cdef double fac, r2, dr[3]

        # ignore self interaction
        if(node.flags & LEAF):
            inside = 1
            for i in range(self.dim):
                if self.x[i][self.current] < node.center[i] - 0.5*node.width:
                    inside = 0
                    break
                if self.x[i][self.current] > node.center[i] + 0.5*node.width:
                    inside = 0
                    break
            if inside:
                return

        # distance between particle and center of mass
        r2 = 0.
        for i in range(self.dim):
            dr[i] = node.group.data.x[i] - self.x[i][self.current]
            r2 += dr[i]**2

        fac = node.group.data.mass / (sqrt(r2) * r2)
        for i in range(self.dim):
            self.a[i][self.current] += fac * dr[i]

    cdef void initialize_particles(self, CarrayContainer pc):
        self.num_particles = pc.get_number_of_items()
        self.current = -1

        self.tags = pc.get_carray('tag')
        pc.pointer_groups(self.a, pc.named_groups['acceleration'])
        pc.pointer_groups(self.x, pc.named_groups['position'])

        # setup information for opening nodes and
        # first particle to process
        self.splitter.initialize_particles(pc)

    cdef int process_particle(self):

        self.current += 1
        if self.current < self.num_particles:
            # skip ghost particles
            while(self.tags[self.current] == Ghost):
                if self.current + 1 < self.num_particles:
                    self.current += 1
                else:
                    return 0

        if self.current < self.num_particles:
            # setup particle for walk
            for i in range(self.dim):
                self.a[i][self.current] = 0.
            self.splitter.process_particle(self.current)
            return 1
        else:
            return 0

