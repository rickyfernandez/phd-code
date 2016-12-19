cimport numpy as np
cimport libc.stdlib as stdlib

from libc.math cimport sqrt

from ..utils.particle_tags import ParticleTAGS
from ..utils.carray cimport DoubleArray, IntArray

cdef int NOT_EXIST = -1
cdef int NOT_LEAF  = 0
cdef int LEAF      = 1

cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost


cdef class GravityNodePool:

    def __init__(self, int num_nodes):
        self.node_array = <Node*> stdlib.malloc(num_nodes*sizeof(Node))
        if self.node_array == NULL:
            raise MemoryError()
        self.used = 0
        self.capacity = num_nodes

    cdef Node* get(self, int count) nogil:
        """
        Allocate 'count' number of nodes from the pool and return pointer to the first node.

        Parameters
        ----------
        int : count
            Number of nodes to allocate
        """
        cdef Node* first_node
        cdef int current = self.used

        self.used += count
        if self.used >= self.capacity:
            self.resize(2*self.capacity)
        first_node = &self.node_array[current]
        return first_node

    cdef void resize(self, int size) nogil:
        """
        Resize the memory pool to have 'size' number of nodes available

        Parameters
        ----------
        int : size
            Number of nodes allocated
        """
        cdef void* node_array = NULL
        if size > self.capacity:
            node_array = <Node*> stdlib.realloc(self.node_array, size*sizeof(Node))

            if node_array == NULL:
                stdlib.free(<void*>self.node_array)
                with gil:
                    raise MemoryError

            self.node_array = <Node*> node_array
            self.capacity = size

    cdef void reset(self):
        """
        Reset the pool
        """
        self.used = 0

    cpdef int number_leaves(self):
        """
        Return number of nodes used from the pool that are leaves.
        """
        cdef int i, num_leaves = 0
        for i in range(self.used):
            if self.node_array[i].leaf == LEAF:
                num_leaves += 1
        return num_leaves

    cpdef int number_nodes(self):
        """
        Return number of nodes used from the pool.
        """
        return self.used

    def __dealloc__(self):
        stdlib.free(<void*>self.node_array)


cdef class GravityTree:

    def __init__(self, DomainLimits domain):
        self.dim = domain.dim
        self.number_nodes = 2**self.dim
        self.domain = domain
        self.nodes = GravityNodePool(1000)

    cdef inline int get_index(self, Node* node, Particle* p) nogil:
        """
        Return child index that particle lives in
        """
        cdef int i, index = 0
        for i in range(self.dim):
            if(p.x[i] > node.center[i]):
                index += (1 << i)
        return index

    cdef inline Node* create_child(self, Node* parent, int index) nogil:
        cdef int i
        cdef Node* child
        cdef double width

        # store particle here
        child = self.nodes.get(1)
        parent.children[index] = self.nodes.used - 1

        for i in range(self.number_nodes):
            child.children[i] = NOT_EXIST
        child.leaf = LEAF

        width = .5*parent.width
        child.width = width
        for i in range(self.dim):
            if( (index >> i) & 1):
                child.center[i] = parent.center[i] + .5*width
            else:
                child.center[i] = parent.center[i] - .5*width

        return child

    def build_tree(self, CarrayContainer pc):
        self._build_tree(pc)

    cdef void _build_tree(self, CarrayContainer pc):
        """
        Build local gravity tree by inserting real particles.
        This method is non-recursive and only adds real particles.
        """
        cdef DoubleArray mass = pc.get_carray('mass')
        cdef IntArray tags = pc.get_carray('tag')

        cdef double width
        cdef int index, current
        cdef Node *node, *child

        cdef int i, j
        cdef Particle p, p2
        cdef np.float64_t *x[3]

        pc.pointer_groups(x, pc.named_groups['position'])

        # create root node
        self.root = self.nodes.get(1)
        self.root.leaf = NOT_LEAF

        # set root center and width
        self.root.width = self.domain.max_length
        for i in range(self.dim):
            self.root.center[i] = .5*\
                    (self.domain.bounds[1][i] - self.domain.bounds[0][i])

        # set children to null
        for j in range(self.number_nodes):
            self.root.children[j] = NOT_EXIST

        # add real particles to the tree
        for i in range(pc.get_number_of_items()):
            if tags.data[i] == Real:

                # copy particle
                p.mass = mass.data[i]
                for j in range(self.dim):
                    p.x[j] = x[j][i]

                # start at root
                current = 0
                while True:

                    node = &self.nodes.node_array[current]
                    if node.leaf != LEAF:

                        index = self.get_index(node, &p)
                        if node.children[index] == NOT_EXIST:

                            # create new child and insert particle
                            child = self.create_child(node, index)
                            child.p.mass = p.mass
                            for j in range(self.dim):
                                child.p.x[j] = p.x[j]

                            # we are done
                            break

                        else:
                            # internal node, travel down
                            current = node.children[index]

                    else:
                        # node is a leaf
                        # we have a particle living here already

                        # copy leaf particle
                        p2.mass = node.p.mass
                        for j in range(self.dim):
                            p2.x[j] = node.p.x[j]

                        # leaf becomes internal node
                        node.leaf = NOT_LEAF

                        # insert previous particle
                        index = self.get_index(node, &p2)
                        child = self.create_child(node, index)

                        # store old particle here
                        child.p.mass = p2.mass
                        for j in range(self.dim):
                            child.p.x[j] = p2.x[j]

                        # try again to insert original particle

        self._update_moments(0, -1)

    cdef void _update_moments(self, int current, int sibling) nogil:
        """
        Recursively update moments of each local node. As a by
        product we collect the first child and sibling of each
        node. This in turn allows for efficient tree walking.
        """
        cdef int i, j, sib
        cdef Node *node, *child

        node = &self.nodes.node_array[current]

        if node.leaf == NOT_LEAF: # internal node

            # clear out node mass, center of mass
            for i in range(self.dim):
                node.p.x[i] = 0.
            node.p.mass = 0.

            # sum moments from each child
            for i in range(self.number_nodes):
                if(node.children[i] != NOT_EXIST):

                    # find sibling of child 
                    j = i + 1
                    while(j < self.number_nodes and node.children[j] == NOT_EXIST):
                        j = j + 1

                    if(j < self.number_nodes):
                        sib = node.children[j]
                    else:
                        sib = sibling

                    self._update_moments(node.children[i], sib)

                    # update node moments
                    child = &self.nodes.node_array[node.children[i]]
                    node.p.mass += child.p.mass
                    for j in range(self.dim):
                        node.p.x[j] += child.p.mass*child.p.x[j]

            # find first child of node
            j = 0
            while(j < self.number_nodes and node.children[j] == NOT_EXIST):
                j = j + 1

            # no longer need children array
            node.first_child = node.children[j]
            node.next_sibling = sibling

            # guard against if node has pseudo particles
            if(node.p.mass):
                for j in range(self.dim):
                    node.p.x[j] /= node.p.mass
        else:
            node.next_sibling = sibling

    def walk(self, Interaction interact, CarrayContainer pc):
        self._walk(interact, pc)

    cdef void _walk(self, Interaction interaction, CarrayContainer pc):
        """
        Walk the tree calculating interactions. Interactions can be any
        calculation between particles.
        """
        cdef int index
        cdef Node *node

        # set particles for loop
        interaction.initialize_particles(pc)

        # loop through each real praticle
        while(interaction.process_particle()):

            # start at first child of root
            index = 0
            node = &self.nodes.node_array[index]
            index = node.first_child

            while(index != -1):

                node = &self.nodes.node_array[index]
                if(node.leaf != LEAF):
                    if(interaction.splitter.split(node)):
                        # node opened travel down
                        index = node.first_child
                    else:
                        # interaction: node-particle
                        interaction.interact(node)
                        index = node.next_sibling
                else:
                    # interaction: particle-particle
                    interaction.interact(node)
                    index = node.next_sibling

    # temporary function to do outputs in python
    def dump_data(self):
        cdef list data_list = []
        cdef Node* node

        cdef int i
        for i in range(self.nodes.used):
            node = &self.nodes.node_array[i]
            if node.leaf == LEAF:
                data_list.append([
                    node.center[0],
                    node.center[1],
                    node.center[2],
                    node.p.mass,
                    node.p.x[0],
                    node.p.x[1],
                    node.width])

        return data_list

    def dump_all_data(self):
        cdef list data_list = []
        cdef Node* node

        cdef int i
        for i in range(self.nodes.used):
            node = &self.nodes.node_array[i]
            data_list.append([
                node.center[0],
                node.center[1],
                node.center[2],
                node.p.mass,
                node.p.x[0],
                node.p.x[1],
                node.leaf,
                node.width])

        return data_list

cdef class Splitter:

    cdef void initialize_particles(self, CarrayContainer pc):
        msg = "Splitter::initialize_particles called!"
        raise NotImplementedError(msg)

    cdef void process_particle(self, long idp):
        self.idp = idp

    cdef int split(self, Node *node):
        msg = "Splitter::split called!"
        raise NotImplementedError(msg)

cdef class BarnesHut(Splitter):
    def __init__(self, double open_angle):
        self.open_angle = open_angle

    cdef void initialize_particles(self, CarrayContainer pc):
        pc.pointer_groups(self.x, pc.named_groups['position'])

    cdef int split(self, Node* node):
        """
        Test if node needs to be open using the Barnes and Hut Criteria.
        """
        cdef int i
        cdef double r2 = 0.

        for i in range(self.dim):
            r2 += (self.x[i][self.idp] - node.p.x[i])**2

        if(node.width*node.width >= r2*self.open_angle*self.open_angle):
            return 1
        else:
            return 0

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
        if node.leaf == LEAF:
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
            dr[i] = node.p.x[i] - self.x[i][self.current]
            r2 += dr[i]**2

        fac = node.p.mass / (sqrt(r2) * r2)
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


#cdef class Nbody:
#    """
#    test class for simple nbody simulation
#    """
#    def __init__(self):
#        pass
#
#    def add_particles(self, cl):
#        cdef int num
#        cdef str axis
#        cdef str dimension = 'xyz'[:self.dim]
#        cdef list acc = []
#
#        if isinstance(cl, phd.CarrayContainer):
#            self.pc = cl
#
#            # modify container to hold gravity data
#            num = self.pc.get_number_of_items()
#            for axis in dimension:
#                self.pc.register_property(num, 'acceleration-' + axis, 'double')
#                acc.append('acceleration-' + axis)
#            self.pc.named_groups['acceleration'] = acc
#
#        else:
#            raise RuntimeError("%s component not type CarrayContainer" % cl.__class__.__name__)
#
#    def calculate_acceleration(self):
#        pass
#
#    def integrate(self, CarrayContainer pc):
#        cdef DoubleArray mass = pc.get_carray('mass')
#        cdef IntArray tags = pc.get_carray('tag')
#
#        cdef int i, j
#        cdef np.float64_t *x[3], *v[3]
#
#        pc.pointer_groups(x, pc.named_groups['position'])
#        pc.pointer_groups(v, pc.named_groups['velocity'])
#
#        for i in range(pc.get_number_of_items()):
#            if tags.data[i] == Real:
#
#                # kick
#                for j in range(self.dim):
#                v[j][i] += 0.5*dt*a[j][i]
#
#                # drift
#                for j in range(self.dim):
#                x[j][i] += dt*v[j][i]
#
#        # calcualte new acceleration
#        # with drift positions
#        self.force.tree.build_tree(self.pc)
#        self.force.walk()
#
#        for i in range(pc.get_number_of_items()):
#            if tags.data[i] == Real:
#
#                # kick
#                for j in range(self.dim):
#                v[j][i] += 0.5*dt*a[j][i]
#
#    def loop(self):
#
#        cdef double time = 0.
#        cdef GravityForce self.force = GravityForce()
#
#        # calculate potential energy
#        self.force.set_splitter(BarnesHut())
#        self.force.walk()
#
#        # use acceleration criteria now since we have
#        # accelerations calculated
#        self.force.set_splitter(AccelerationCriteria())
#        self.force.walk()
#
#        while time < self.total_time:
#            self.output_data()
#            self.vis()
#            self.integrate()
#            time += self.dt


#cdef class AccelerationCriteria(splitter):
#    cdef bool split(self, node* node, particle *p) nogil:
#        cdef int i
#        cdef double r2 = 0.
#
#        # ignore self interaction
#        if(inside_box(node, p.x)):
#            return False
#
#        for i in range(self.dim):
#            r2 += (p.x[i] - node.p.x[i])**2
#
#        if(node.width*node.width*node.p.mass <= r2*r2*p.old_a):
#            return True
#        else:
#            return False
#

#cdef class GravityPotential(Interaction):
#    cdef inline interact(self, Node* node, Particle *p) nogil:
#        double r2
#
#        r2 = 0.
#        for i in range(self.dim):
#            r2 += (p.x[i] - node.p.x[i])**2
#
#        for i in range(self.dim):
#            p.a[i] -= node.p.mass / sqrt(r2)


# everything below here is experimental
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#cdef class GlobalTree:

#    def __init__(self, LoadBalance ld):
#        self.tree = tree
#        self.node_pool = TreeMemoryPool(0)
#        self.pseudo_nodes = LongArray()

#    def construct_tree(self):
#        cdef long i, my_proc
#        cdef long num_nodes = self.tree.used
#
#        # first copy load blance tree
#        self.mem_pool.resize(num_nodes)
#        for i in range(num_nodes):
#
#            node = self.tree.mem_pool[i]
#            grav_node = self.mem_pool[i]
#
#            # store position of pseudo node
#            if leaf_pid[node.array_index] != rank:
#                self.pseudo_nodes.append(i)
#                grav_node.pseudo = 1
#
#            # copy information
#            grav_node.box_length = node.box_length
#
#            for j in range(dim):
#                grav_node.center[j] = node.center[j]










                        # put old
#
#    cdef inline void create_children(self, Node* node) nogil:
#        cdef int i, j
#        cdef Node* child
#        cdef Node* new_node = self.mem_pool.get(self.num_children)
#
#        # displacement to child
#        node.children_start = new_node - node
#
#        for i in range(self.num_children):
#
#            child = self.get_child(node, i)
#
#            child.width = .5*node.width
#            child.children_start = LEAF
#            child.has_particle = NO_PARTICLE
#
#            for j in range(self.dim):
#                if( (i >> j) & 1):
#                    child.center[j] = node.center[j] + width
#                else:
#                    child.center[j] = node.center[j] - width
#
#    cdef void update_node(self, Node* node) nogil:
#        cdef int i, j
#        cdef Node* child
#
#        if(node.children_start != LEAF):
#
#            node.p.mass = 0.
#            for i in range(self.dim):
#                node.p.x[i] = 0.
#
#            for j in range(self.num_children):
#                child = node.get_child(j)
#
#                if(child.children_start != LEAF):
#                    self.update_node(child)
#
#                node.p.mass += child.p.mass
#                for j in range(self.dim):
#                    node.p.x[i] = child.p.x[i]*child.p.mass
#
#            for j in range(self.dim):
#                node.p.x[j] /= node.p.mass
#
#        else:
#            if(node.has_particle == NO_PARTICLE):
#                node.p.mass = 1.
#                for j in range(self.xim):
#                    node.p.x[j] = 0.
#
#    cdef inline Node* get_child(self, Node* node, int index) nogil:
#        """
#        Return child from index
#        """
#        return node + node.children_start + index
#
#    cdef void build_tree(self, CarrayContainer pc) nogil:
#        """
#        build local gravity tree
#        """
#        cdef DoubleArray mass = pc.get_carray('mass')
#        cdef IntArray tags = pc.get_carray('tag')
#
#        cdef int i, j
#        cdef Particle p
#        cdef np.float64_t *x[3]
#
#        pc.pointer_groups(x, pc.named_groups['position'])
#
#        # copy domain information to root node
#        self.root = self.pool.get(1)
#
#        self.root.width = .5*self.domain.max_length
#        for i in range(self.dim):
#            self.root.center[i] = .5*\
#                    (self.domain.bounds[1][i] - self.domain.bounds[0][i])
#
#        for i in range(self.number_nodes):
#            self.root.children[i] = NOT_EXIST
#
#        for i range(pc.get_number_of_items()):
#            if tags.data[i] == Real:
#
#                for j in range(dim):
#                    p.x[j] = x[j][i]
#                p.mass = mass.data[i]
#
#                self.add_particle(self.root, &p)
#
#        self.update_moments(0, -1)
#    cdef void add_particle(Node* node, Particle* p) nogil:
#        """
#        Add particle to the tree
#        """
#        cdef int i, j, index
#        cdef Particle p2
#        cdef Node* child
#
#        index = self.get_index(node, p)
#
#        # empty node, insert particle
#        if node.children[index] == NOT_EXIST:
#
#            # create child
#            child = self.nodes.get(1)
#            node.children[index] = self.nodes.used - 1
#
#            # set pointers of children to null
#            for j in range(self.number_children):
#                child.children[j] = NOT_EXIST
#
#            # pass information from parent
#            # and new particle
#            width = .5*node.width
#            child.width = width
#
#            for j in range(self.dim):
#
#                # particle coordinates
#                child.p.x[j] = p.x[j]
#
#                # center coordinates of new child
#                if( (index >> j) & 1):
#                    child.center[j] = node.center[j] + .5*width
#                else:
#                    child.center[j] = node.center[j] - .5*width
#
#            # particle mass
#            child.p.mass = p.mass
#            child.leaf = LEAF
#
#        # non-empty node, subdivide, insert and try again 
#        else:
#
#            # extract child
#            child = self.nodes[node.children[index]]
#
#            if child.leaf == LEAF:
#
#                # remove old particle 
#                p2.mass = child.p.mass
#                for i in range(self.dim):
#                    p2.x[j] = child.p.x[j]
#                child.leaf == NOT_LEAF
#
#                # add old particle back
#                self.add_particle(child, &p2)
#                self.add_particle(child, p)
#
#            else:
#                self.add_particle(child, p)
#
#    def dump_data(self, CarrayContainer pc):
#        cdef IntArray tags = pc.get_carray('tag')
#
#        cdef int i, index
#        cdef int current, parent
#        cdef list data_list = []
#
#        cdef Node *node, *father
#        cdef Particle p
#        cdef np.float64_t *x[3], center[3]
#
#        pc.pointer_groups(x, pc.named_groups['position'])
#
#        for i in range(pc.get_number_of_items()):
#            if tags.data[i] == Real:
#
#                for j in range(self.dim):
#                    p.x[j] = x[j][i]
#
#                # start at the root
#                current = 0
#                while True:
#
#                    node = &self.nodes.node_array[current]
#                    print 'current:', current, 'leaf:', node.leaf
#
#                    if node.leaf == LEAF:
#
#                        father = &self.nodes.node_array[parent]
#                        width = .5*father.width
#
#                        # create center coordinates for leaf
#                        for j in range(self.dim):
#                            if( (index >> j) & 1):
#                                center[j] = father.center[j] + .5*width
#                            else:
#                                center[j] = father.center[j] - .5*width
#
#                        data_list.append([
#                            center[0],
#                            center[1],
#                            center[2],
#                            width])
#
#                        # done with this particle
#                        break
#
#                    else:
#
#                        # internal node, travel down
#                        index = self.get_index(node, &p)
#
#                        parent = current
#                        current = node.children[index]
#                        print 'travel down', parent, current
#
#        return data_list

