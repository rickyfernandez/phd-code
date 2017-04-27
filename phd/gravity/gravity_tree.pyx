import numpy as np
cimport numpy as np
cimport libc.stdlib as stdlib

from mpi4py import MPI

from libc.math cimport sqrt

from ..utils.particle_tags import ParticleTAGS
from ..utils.carray cimport DoubleArray, IntArray, LongArray, LongLongArray
from ..load_balance.tree cimport TreeMemoryPool as Pool


# tree flags
cdef int NOT_EXIST = -1
cdef int ROOT = 0
cdef int ROOT_SIBLING = -1
cdef int LEAF = 0x01
cdef int HAS_PARTICLE = 0x02
cdef int TOP_TREE = 0x04
cdef int TOP_TREE_LEAF = 0x08
cdef int TOP_TREE_LEAF_REMOTE = 0x10

cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost


cdef class GravityNodePool:

    def __init__(self, int num_nodes):
        self.node_array = <Node*> stdlib.malloc(num_nodes*sizeof(Node))
        if self.node_array == NULL:
            raise MemoryError()
        self.used = 0
        self.capacity = num_nodes

    cdef Node* get(self, int count):
        """
        Allocate count number of nodes from the pool and return
        pointer to the first node.

        Parameters
        ----------
        int : count
            Number of nodes to allocate

        Returns
        -------
        Node*
            Pointer to node allocated
        """
        cdef Node* first_node
        cdef int current = self.used

        if (self.used + count) > self.capacity:
            self.resize(2*self.capacity)
        first_node = &self.node_array[current]
        self.used += count

        return first_node

    cdef void resize(self, int size):
        """
        Resize the memory pool to have size number of nodes available
        for use. Note this does not mean there are size nodes used.

        Parameters
        ----------
        int : size
            Number of nodes allocated
        """
        cdef void* node_array = NULL
        if size > self.capacity:
            node_array = <Node*>stdlib.realloc(self.node_array, size*sizeof(Node))

            if node_array ==  NULL:
                stdlib.free(<void*>self.node_array)
                raise MemoryError('Insufficient Memory in gravity pool')

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

cdef class GravityTree:
    """
    Solves gravity by barnes and hut algorithm in serial or
    parallel. Should only be used for parallel runs. This
    algorithm heavily depends on the LoadBalance class.
    The algorithm works in 2d or 3d.
    """
    def __init__(self, int parallel=0):

        #self.domain = domain
        #self.load_bal = load_bal
        self.parallel = parallel

    def _initialize(self):
        '''
        Initialize variables for the gravity tree. Tree pool and
        remote nodes are allocated as well dimension of the tree.
        '''
        cdef str axis
        cdef dict state_vars = {}
        cdef dict named_groups = {}

        self.dim = self.domain.dim
        self.number_nodes = 2**self.dim
        self.nodes = GravityNodePool(1000)

        if self.parallel:

            self.node_disp = np.zeros(self.load_bal.size, dtype=np.int32)
            self.node_counts = np.zeros(self.load_bal.size, dtype=np.int32)

            self.rank = self.load_bal.rank
            self.size = self.load_bal.size

            self.send_counts = np.zeros(self.size, dtype=np.int32)
            self.recv_counts = np.zeros(self.size, dtype=np.int32)

            state_vars['map']  = 'long'
            state_vars['proc'] = 'long'
            state_vars['mass'] = 'double'

            named_groups['com'] = []
            for axis in 'xyz'[:self.dim]:
                state_vars['com-' + axis] = 'double'
                named_groups['com'].append('com-' + axis)
            named_groups['moments'] = ['mass'] + named_groups['com']

            self.remote_nodes = CarrayContainer(var_dict=state_vars)
            self.remote_nodes.named_groups = named_groups

            self.buffer_id = IntArray()
            self.buffer_pid = IntArray()

            self.buffer_import = CarrayContainer()
            self.buffer_export = CarrayContainer()

            self.import_interaction # add logic for assigment
            self.local_interaction

    cdef inline int get_index(self, int parent_index, np.float64_t x[3]):
        """
        Return index of child from parent node with node_index. Children
        are laid out in z-order.

        Parameters
        ----------
        node_index : int
            index of node that you want to find child of
        x : np.float64_t array
            particle coordinates to find child

        Returns
        -------
        int
            integer of child laid in z-order
        """
        cdef int i, index = 0
        cdef Node* node = &self.nodes.node_array[parent_index]
        for i in range(self.dim):
            if(x[i] > node.center[i]):
                index += (1 << i)
        return index

    cdef inline Node* create_child(self, int parent_index, int child_index):
        '''
        Create child node given parent index and child index. Note parent_index
        refers to memory pool and child_index refers to [0,3] in 2d or [0,7]
        for children array in parent.

        Parameters
        ----------
        parent_index : int
            index of parent in pool to subdivide
        child_index : int
            index of child relative to parent.children array

        Returns
        -------
        child : Node*
            child pointer
        '''
        cdef int i
        cdef Node *child, *parent
        cdef double width

        # allocate child
        child = self.nodes.get(1)

        # pass parent info to child
        parent = &self.nodes.node_array[parent_index]
        parent.children[child_index] = self.nodes.used - 1

        # parent no longer leaf
        parent.flags &= ~LEAF

        for i in range(self.number_nodes):
            child.u.children[i] = NOT_EXIST

        child.flags = LEAF
        width = .5*parent.width
        child.width = width

        for i in range(self.dim):
            # create center coords for child
            if( (child_index >> i) & 1):
                child.center[i] = parent.center[i] + .5*width
            else:
                child.center[i] = parent.center[i] - .5*width

        return child

    cdef inline void create_children(self, int parent_index):
        """
        Given a parent node, subdivide domain into (4-2d, 8-3d) children. The algorithm
        is independent of dimension.
        """
        cdef double width
        cdef Node *child, *parent
        cdef int i, k, start_index

        # create a block of children
        child = self.nodes.get(self.number_nodes)          # reference of first child
        start_index = self.nodes.used - self.number_nodes  # index of first child

        parent = &self.nodes.node_array[parent_index]
        width = .5*parent.width                            # box width of children
        parent.flags &= ~LEAF                              # parent no longer leaf

        # loop over each child and pass parent information
        for i in range(self.number_nodes):

            # store child index in node array
            parent.u.children[i] = start_index + i
            child = &self.nodes.node_array[start_index + i]

            child.flags = LEAF
            child.width = width

            # set children of children to null
            for k in range(self.number_nodes):
                child.u.children[k] = NOT_EXIST

            # create center coordinates from parent
            # children are put in z-order
            for k in range(self.dim):
                # create center coords for child
                if((i >> k) & 1):
                    child.center[k] = parent.center[k] + .5*width
                else:
                    child.center[k] = parent.center[k] - .5*width

    cdef void _build_top_tree(self):
        """
        Copy the load balance tree. The tree is the starting point to add
        particles since this tree is common to all processors. Note the
        load balance tree is in hilbert order.
        """
        cdef int i
        cdef np.int32_t *node_ind

        cdef Node *node
        cdef LoadNode *load_root = self.load_bal.tree.root

        cdef Pool pool = self.load_bal.tree.mem_pool

        cdef LongArray leaf_pid = self.load_bal.leaf_pid
        cdef LongArray maps = self.remote_nodes.get_carray('map')
        cdef LongArray proc = self.remote_nodes.get_carray('proc')

        # resize memory pool to hold tree - this only allocates available
        # memory it does not create nodes
        self.nodes.resize(pool.number_nodes())

        # resize container to hold leaf data 
        self.remote_nodes.resize(pool.number_leaves())

        # copy global partial tree in z-order, collect leaf index for mapping 
        self._create_top_tree(ROOT, load_root, maps.get_data_ptr())

        # remote nodes are in load balance order, hilbert and processor, this
        # allows for easy communication. Transfer processor id
        for i in range(self.size):
            self.node_counts[i] = 0

        for i in range(leaf_pid.length):
            proc.data[i] = leaf_pid.data[i]
            self.node_counts[leaf_pid.data[i]] += 1

        self.node_disp[0] = 0
        for i in range(1, self.size):
            self.node_disp[i] = self.node_counts[i-1] + self.node_disp[i-1]

    cdef void _create_top_tree(self, int node_index, LoadNode* load_parent,
            np.int32_t* node_map):
        """
        Copys the load balance tree. The tree is the starting point to add
        particles since this tree is common to all processors. Note the
        load balance tree is in hilbert order, so care is taken to put
        the gravity tree in z-order. Note the leafs of the top tree are
        the objects used for the load balance. The leafs are stored in
        the remote_nodes container and are in hilbert and processor order.
        The map array is used to map from remote_nodes to nodes in the gravity
        tree. All nodes will be labeled to belong to the top tree.

        Parameters
        ----------
        node_index : int
            index of node in gravity tree
        load_parent : LoadNode*
            node pointer to load balance tree
        node_map : np.int32_t*
            array to map remote nodes to leafs in gravity tree
        """
        cdef int index, i
        cdef Node* parent = &self.nodes.node_array[node_index]

        # label node in top tree
        parent.flags |= TOP_TREE

        if load_parent.children_start == -1: # leaf stop
            parent.flags |= TOP_TREE_LEAF
            node_map[load_parent.array_index] = node_index
        else: # non leaf copy

            # create children in z-order
            self.create_children(node_index)

            # create children could of realloc
            parent = &self.nodes.node_array[node_index]

            # travel down to children
            for i in range(self.number_nodes):

                # grab next child in z-order
                index = load_parent.zorder_to_hilbert[i]
                self._create_top_tree(
                        parent.children[i], load_parent + load_parent.children_start + index,
                        node_map)

    cdef int _leaf_index_toptree(self, np.int64_t key):
        """
        Find index of local tree which coincides with given key
        inside leaf in top tree.
        """
        cdef LoadNode* load_node
        cdef LongArray maps = self.remote_nodes.get_carray('map')

        load_node = self.load_bal.tree.find_leaf(key)
        return maps.data[load_node.array_index]

    cdef void _create_root(self):
        cdef int k
        cdef Node *root

        # clear out node pool 
        self.nodes.reset()

        # create root with domain information
        root = self.nodes.get(1)
        root.flags = LEAF
        root.width = self.domain.max_length
        for k in range(self.dim):
            root.center[k] = .5*\
                    (self.domain.bounds[1][k] - self.domain.bounds[0][k])

        # set root children to null
        for k in range(self.number_nodes):
            root.u.children[k] = NOT_EXIST

    def _build_tree(self, CarrayContainer pc):
        """
        Build local gravity tree by inserting real particles.
        This method is non-recursive and only adds real particles.
        Note, leaf nodes may have a particle. The distinction is for
        parallel tree builds because the top tree will have leafs
        without any particles.
        """
        cdef IntArray tags = pc.get_carray('tag')
        cdef DoubleArray mass = pc.get_carray('mass')
        cdef LongLongArray keys = pc.get_carray('key')

        cdef double width
        cdef int index, current
        cdef Node *node, *child

        cdef int i, j, k
        cdef double xi[3], xj[3]

        # get pointer to particle position and mass
        pc.pointer_groups(self.x, pc.named_groups['position'])
        self.m = mass.get_data_ptr()

        # reset tree if needed and create root node
        self._create_root()

        # create top tree if parallel run
        if self.parallel:
            self._build_top_tree()

        # add real particles to tree
        for i in range(pc.get_number_of_items()):
            if tags.data[i] == Real:

                # copy particle
                for k in range(self.dim):
                    xi[k] = self.x[k][i]

                if self.parallel:
                    # start at top tree leaf
                    current = self._leaf_index_toptree(keys.data[i])
                else:
                    # start at root
                    current = ROOT

                while True:
                    node = &self.nodes.node_array[current]
                    if (node.flag & LEAF):
                        if (node.flag & HAS_PARTICLE):

                            # particle living here already
                            # copy particle 
                            pj_index = node.u.n.pid
                            for k in range(self.dim):
                                xj[k] = self.x[k][pj_index]

                            # reset children to null due to union
                            for k in range(self.number_nodes):
                                node.u.children[k] = NOT_EXIST

                            # node becomes internal node
                            node.flags &= ~(LEAF|HAS_PARTICLE)

                            # create child to store previous particle
                            # children in parent in union is used
                            index = self.get_index(current, xj)
                            child = self.create_child(current, index)

                            # store old particle here
                            child.flag |= (LEAF|HAS_PARTICLE)
                            child.u.n.pid = j

                            # try to insert original particle again

                        else:
                            # store particle here
                            node.flags |= HAS_PARTICLE
                            # overwrites children in union
                            node.u.n.pid = i
                            break

                    else:
                        # find child to store particle
                        index = self.get_index(current, xi)

                        # if child does not exist in this slot
                        # create child and store particle
                        if node.u.children[index] == NOT_EXIST:
                            child = self.create_child(current, index)
                            child.flags |= (LEAF|HAS_PARTICLE)
                            child.u.n.pid = i
                            break # particle done

                        else: # internal node, travel down
                            current = node.u.children[index]

        self._update_moments(ROOT, ROOT_SIBLING)

        if self.parallel:
            self._export_import_remote_nodes()

        #root = &self.nodes.node_array[0]
        #print 'rank=', self.rank, 'total root mass:', root.p.mass

    cdef void _update_moments(self, int current, int sibling):
        """
        Recursively update moments of each local node. As a by
        product we collect the first child and sibling of each
        node, which allows for efficient tree walking.
        """
        cdef int i, j, sib, pid
        cdef Node *node, *child
        cdef double mass, com[3]

        node = &self.nodes.node_array[current]

        # temp variables because we cant over
        # write children yet in union
        mass = 0.
        for i in range(self.dim):
            com[i] = 0.

        if((node.flags & LEAF) != LEAF): # internal node

            # sum moments from each child
            for i in range(self.number_nodes):
                if(node.u.children[i] != NOT_EXIST):

                    # find sibling of child 
                    j = i + 1
                    while(j < self.number_nodes and node.u.children[j] == NOT_EXIST):
                        j = j + 1

                    if(j < self.number_nodes):
                        sib = node.u.children[j]
                    else:
                        sib = sibling

                    self._update_moments(node.u.children[i], sib)

                    # update node moments
                    child = &self.nodes.node_array[node.u.children[i]]
                    mass += child.u.n.mass
                    for j in range(self.dim):
                        com[j] += child.u.n.mass*child.u.n.com[j]

            # find first child of node
            j = 0
            while(j < self.number_nodes and node.u.children[j] == NOT_EXIST):
                j = j + 1

            # no longer need children array in union
            node.u.n.first_child = node.u.children[j]
            node.u.n.next_sibling = sibling

            node.u.n.mass = mass
            if(mass):
                for j in range(self.dim):
                    node.u.n.com[j] = com[j]/mass
            else:
                for j in range(self.dim):
                    node.u.n.com[j] = 0.

        else:

            # no longer need children array
            node.u.n.first_child  = NOT_EXIST
            node.u.n.next_sibling = sibling

            node.u.n.mass = mass
            for j in range(self.dim):
                node.u.n.com[j] = com[j]

            # remote leafs may not have particles
            if (node.flags & HAS_PARTICLE):
                pid = node.u.n.pid

                # copy particle information
                node.u.n.mass = self.m[pid]
                for j in range(self.dim):
                    node.u.n.com[j] = self.x[pid]

    cdef void _update_remote_moments(self, int current):
        """
        Recursively update moments of each local node. As a by
        product we collect the first child and sibling of each
        node. This in turn allows for efficient tree walking.
        """
        cdef int i, j, index
        cdef Node *node, *child
        cdef double mass, com[3]

        node = &self.nodes.node_array[current]

        # check if node is not a top tree leaf
        if((node.flags & TOP_TREE_LEAF) != TOP_TREE_LEAF):

            mass = 0.
            for i in range(self.dim):
                com[i] = 0.

            # sum moments from each child
            index = node.u.n.first_child
            while(index != node.u.n.sibling):

                # update node moments
                child = &self.nodes.node_array[index]
                self._update_remote_moments(index)

                mass += child.u.n.mass
                for j in range(self.dim):
                    com[j] += child.u.n.mass*child.u.n.com[j]
                index = child.u.n.sibling

            if(mass):
                for j in range(self.dim):
                    com[j] /= mass

            node.u.n.mass = mass
            for j in range(self.dim):
                node.u.n.com[j] = com[j]

#    cdef void _export_import_remote_nodes(self):
#        cdef int i, j
#        cdef Node *node
#        cdef np.float64_t* comx[3]
#
#        cdef LongArray proc   = self.remote_nodes.get_carray('proc')
#        cdef LongArray maps   = self.remote_nodes.get_carray('map')
#        cdef DoubleArray mass = self.remote_nodes.get_carray('mass')
#
#        self.remote_nodes.pointer_groups(comx, self.remote_nodes.named_groups['com'])
#
#        # collect moments belonging to current processor
#        for i in range(self.remote_nodes.get_number_of_items()):
#            if proc.data[i] == self.rank:
#
#                # copy moment information
#                node = &self.nodes.node_array[maps.data[i]]
#                for j in range(self.dim):
#                    comx[j][i] = node.p.x[j]
#                mass.data[i] = node.p.mass
#
#        # export local node info and import remote node
#        for field in self.remote_nodes.named_groups['moments']:
#            self.load_bal.comm.Allgatherv(MPI.IN_PLACE,
#                    [self.remote_nodes[field], self.node_counts, self.node_disp, MPI.DOUBLE])
#
#        # copy remote nodes to tree
#        for i in range(self.remote_nodes.get_number_of_items()):
#            if proc.data[i] != self.rank:
#
#                node = &self.nodes.node_array[maps.data[i]]
#                for j in range(self.dim):
#                    node.p.x[j] = comx[j][i]
#                node.p.mass = mass.data[i]
#
#        # update moments
#        self._update_remote_moments(0)
#
#    def walk(self, Interaction interact, CarrayContainer pc):
#        if self.parallel:
#            self._parallel_walk(interact, pc)
#        else:
#            self._serial_walk(interact, pc)
#
#    cdef void _serial_walk(self, Interaction interaction, CarrayContainer pc):
#        """
#        Walk the tree calculating interactions. Interactions can be any
#        calculation between particles.
#        """
#        cdef int index
#        cdef Node *node
#
#        # set particles for loop
#        interaction.initialize_particles(pc)
#
#        # loop through each real praticle
#        while(interaction.process_particle()):
#
#            # start at first child of root
#            index = 0
#            node = &self.nodes.node_array[index]
#            index = node.first_child
#
#            while(index != -1):
#
#                node = &self.nodes.node_array[index]
#                if(node.leaf != LEAF):
#                    # should node be opened
#                    if(interaction.splitter.split(node)):
#                        index = node.first_child
#                    else:
#                        # interaction: node-particle
#                        interaction.interact(node)
#                        index = node.next_sibling
#                else:
#                    # interaction: particle-particle
#                    interaction.interact(node)
#                    index = node.next_sibling
#
#    cdef void _local_walk(self, Interaction interaction, CarrayContainer pc, int local):
#        """
#        Walk the tree calculating interactions. Interactions can be any
#        calculation between particles.
#        """
#        cdef int index
#        cdef Node *node
#
#        cdef LongArray pid = self.remote_nodes.get_carray("proc")
#
#        # loop through each real praticle
#        while(interaction.process_particle()):
#
#            # start at root or next node from previous walk
#            if local:
#                index = interaction.start_node_index()
#            else:
#                current = self._leaf_index_toptree(keys.data[i])
#                node = &self.nodes.node_array[current]
#            while(index != -1):
#
#                node = &self.nodes.node_array[index]
#                if(node.leaf != LEAF):
#                    if local:
#                    # should node be opened
#                        if(interaction.splitter.split(node)):
#                            index = node.first_child
#                        else:
#                            # interaction: node-particle
#                            interaction.interact(node)
#                            index = node.next_sibling
#                    else:
#                        # if node does not belong in the tree
#                        if (node.flag & TOP_LEVEL_DEPEND):
#                            index = node.next_sibling
#                else: # leaf can be remote or local
#                    if node.remote:
#                        if local: # for local particles check for export
#                            if(interaction.splitter.split(node)):
#                                # node opend flag for export
#                                self.buffer_pid.append(pid.data[node.index])
#                                self.buffer_id.append(interaction.current)
#
#                                # check if buffer is full, halt walk if true
#                                if self.buffer_pid.length == self.max_export:
#                                    # save node to continue walk
#                                    interaction.particle_not_finished(
#                                            node.next_sibling)
#                                    return # break out of walk
#                                else:
#                                    index = node.next_sibling
#                            else: # interaction: node-particle
#                                interaction.interact(node)
#                                index = node.next_sibling
#                        else: # non-local particle skip node
#                            index = node.next_sibling
#                    else:# interaction: particle-particle
#                        interaction.interact(node)
#                        index = node.next_sibling
#
#    cdef void _parallel_walk(self, CarrayContainer pc):
#        cdef long num_import
#        cdef int local_done, global_done
#
#        cdef np.ndarray loc_done, glb_done
#        cdef np.ndarray buffer_id_npy, buffer_pid_npy
#
#        loc_done = np.zeros(1, dtype=np.int32)
#        glb_done = np.zeros(1, dtype=np.int32)
#
#        # clear out buffers
#        self.buffer_id.reset()
#        self.buffer_pid.reset()
#
#        # convenience arrays
#        buffer_id_npy  = self.buffer_id.get_npy_array()
#        buffer_pid_npy = self.buffer_pid.get_npy_array()
#
#        # setup local particles for walk
#        self.local_interaction.initialize_particles(pc)
#        while True:
#
#            # perform walk while flagging particles for export
#            # once the buffer is full the walk will hault
#            self._local_walk(self.local_interaction, pc, LOCAL)
#
#            # put particles in process order
#            ind = buffer_pid_npy.argsort()
#            buffer_id_npy[:]  = buffer_id_npy[ind]
#            buffer_pid_npy[:] = buffer_pid_npy[ind]
#
#            # count how many particles are going to each process
#            for i in range(self.size):
#                self.send_counts[i] = 0
#            for i in range(self.buffer_pid.length):
#                self.send_counts[self.buffer_pid.data[i]] += 1
#
#            # send number of export to all processors
#            self.load_bal.comm.Alltoall([self.send_counts, MPI.INT],
#                    [self.recv_counts, MPI.INT])
#
#            # how many remote particles are incoming
#            num_import = 0
#            for i in range(self.size):
#                num_import += self.recv_counts[i]
#
#            # create displacement arrays 
#            self.send_disp[0] = self.recv_disp[0] = 0
#            for i in range(1, self.size):
#                self.send_disp[i] = self.send_counts[i-1] + self.send_disp[i-1]
#                self.recv_disp[i] = self.recv_counts[i-1] + self.recv_disp[i-1]
#
#            # copy flagged particles into buffer
#            self.buffer_export.copy(pc,
#                    self.buffer_id_npy,
#                    pc.named_group['gravity-walk-export'])
#
#            # resize to fit number of particle incoming
#            self.import_buffer.resize(num_import)
#
#            # send our particles / recieve particles 
#            exchange_particles(self.buffer_import, self.buffer_export,
#                    self.send_counts, self.recv_conts, self.load_bal.comm,
#                    pc.named_groups['gravity-walk-export'],
#                    self.send_disp, self.recv_disp)
#
#            # walk remote particles
#            self.import_interaction.initialize_particles(self.buffer_import)
#            self._local_walk(self.import_interaction, self.buffer_import, REMOTE)
#
#            # recieve back our paritcles / send back particles
#            exchange_particles(self.buffer_export, self.buffer_import,
#                    self.send_counts, self.recv_counts, self.load_bal.com,
#                    pc.named_groups['gravity-walk-import'],
#                    self.recv_disp, self.send_disp, self.recv_disp)
#
#            # copy back our data
#            pc.insert(self.buffer_export
#                    pc.named_group['gravity-walk-export'],
#                    self.buffer_id_npy)
#
#            # let all processors know if walk is complete 
#            glb_done[0] = 0
#            loc_done[0] = interaction.done_processing()
#            comm.Allreduce([loc_done, MPI.INT], [glb_done, MPI.INT], op=MPI.SUM)
#
#            # if all processors tree walks are done exit
#            if glb_done[0] == self.size:
#                break

    # -- delete later --
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

    # -- delete later --
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

    # -- delete later --
    def dump_remote(self):
        cdef int i, j
        cdef Node* node
        cdef list data_list = []
        cdef LongArray maps = self.remote_nodes.get_carray('map')
        cdef LongArray proc = self.remote_nodes.get_carray('proc')

        print 'rank:', self.load_bal.rank, 'number of nodes:', self.nodes.used

        for i in range(self.remote_nodes.get_number_of_items()):
            j = maps.data[i]
            node = &self.nodes.node_array[j]
            data_list.append([
                node.center[0],
                node.center[1],
                node.center[2],
                node.p.mass,
                node.p.x[0],
                node.p.x[1],
                proc.data[i],
                node.width])

        return data_list
