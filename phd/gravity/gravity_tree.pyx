cimport numpy as np
cimport libc.stdlib as stdlib

from libc.math cimport sqrt

from ..utils.particle_tags import ParticleTAGS
from ..utils.carray cimport DoubleArray, IntArray, LongArray, LongLongArray
from ..load_balance.tree cimport TreeMemoryPool as Pool

# flags
cdef int NOT_EXIST = -1

cdef int NOT_LEAF = 0
cdef int LEAF     = 1

cdef int DEPENDANT     = 1
cdef int NOT_DEPENDANT = 1

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

            state_vars['map']  = 'long'
            state_vars['proc'] = 'long'
            state_vars['mass'] = 'double'

            named_groups['com'] = []
            for axis in 'xyz'[:self.dim]:
                stat_vars['com-' + axis] = 'double'
                named_groups['com'].append('com-' + axis)
            named_groups['moments'] = ['mass'] + named_groups['com']

            self.remote_nodes = CarrayContainer(var_dict=state_vars)
            self.remote_nodes.named_groups = named_groups

    cdef inline int get_index(self, int node_index, Particle* p):
        """
        Return index of child from parent node with node_index. Children
        are laid out in z-order.

        Parameters
        ----------
        node_index : int
            index of node that you want to find child of
        p : Particle*
            particle used to find child

        Returns
        -------
        int
            integer of child laid in z-order
        """
        cdef int i, index = 0
        cdef Node* node = &self.nodes.node_array[node_index]
        for i in range(self.dim):
            if(p.x[i] > node.center[i]):
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
        cdef Node* child
        cdef double width

        # allocate child
        child = self.nodes.get(1)

        # pass parent info to child
        parent = &self.nodes.node_array[parent_index]
        parent.children[child_index] = self.nodes.used - 1

        for i in range(self.number_nodes):
            child.children[i] = NOT_EXIST

        child.leaf = LEAF
        child.dependant = NOT_DEPENDANT

        width = .5*parent.width
        child.width = width
        for i in range(self.dim):
            if( (index >> i) & 1):
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
        parent.leaf = NOT_LEAF                             # parent no longer leaf

        # loop over each child and pass parent information
        for i in range(self.number_nodes):

            # store child index in node array
            parent.children[i] = start_index + i
            child = &self.nodes.node_array[start_index + i]

            child.leaf = LEAF
            child.dependant = NOT_DEPENDANT
            child.width = width

            # set children of children to null
            for k in range(self.number_nodes):
                child.children[k] = NOT_EXIST

            # create center coordinates from parent
            # children are put in z-order
            for k in range(self.dim):
                if((i >> k) & 1):
                    child.center[k] = parent.center[k] + .5*width
                else:
                    child.center[k] = parent.center[k] - .5*width

    #cdef void _build_top_tree(self):
    def _build_top_tree(self):
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
        self._create_top_tree(0, load_root, maps.get_data_ptr())

        # remote nodes are in load balance order, hilbert and processor, this
        # allows for easy communication. Transfer processor id
        for i in range(leaf_pid.length):
            proc.data[i] = leaf_pid.data[i]

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
        tree.

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

        if load_parent.children_start == -1: # leaf stop
            parent.dependant = NOT_DEPENDANT
            node_map[load_parent.array_index] = node_index
        else: # non leaf copy

            # create children in z-order
            parent.dependant = DEPENDANT
            self.create_children(parent)

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

    #cdef void _build_tree(self, CarrayContainer pc):
    def _build_tree(self, CarrayContainer pc):
        """
        Build local gravity tree by inserting real particles.
        This method is non-recursive and only adds real particles.
        """
        cdef IntArray tags = pc.get_carray('tag')
        cdef DoubleArray mass = pc.get_carray('mass')
        cdef LongLongArray keys = pc.get_carray('key')

        cdef double width
        cdef int index, current
        cdef Node *node, *child, *root

        cdef int i, j, tmp
        cdef Particle p, p2
        cdef np.float64_t *x[3]

        pc.pointer_groups(x, pc.named_groups['position'])

        # clear out node pool 
        self.nodes.reset()

        # create root
        root = self.nodes.get(1)
        root.leaf = NOT_LEAF

        # set root center and width
        root.width = self.domain.max_length
        for i in range(self.dim):
            root.center[i] = .5*\
                    (self.domain.bounds[1][i] - self.domain.bounds[0][i])

        # set root children to null
        for j in range(self.number_nodes):
            root.children[j] = NOT_EXIST

        # create top tree if parallel run
        if self.parallel:
            self._build_top_tree()

        # add real particles to tree
        for i in range(pc.get_number_of_items()):
            if tags.data[i] == Real:

                # copy particle
                p.mass = mass.data[i]
                for j in range(self.dim):
                    p.x[j] = x[j][i]

                if self.parallel:
                    # start at leaf in top tree
                    current = self._leaf_index_toptree(keys.data[i])
                    node = &self.nodes.node_array[current]
                    node.leaf = NOT_LEAF
                else:
                    # start at root
                    current = 0

                while True:

                    node = &self.nodes.node_array[current]

                    # check if node is not a leaf
                    if node.leaf != LEAF:
                        # find child to store 
                        index = self.get_index(current, &p)
                        if node.children[index] == NOT_EXIST:

                            child = self.create_child(current, index)
                            child.p.mass = p.mass
                            for j in range(self.dim):
                                child.p.x[j] = p.x[j]
                            break # particle done

                        else: # internal node, travel down
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
                        index = self.get_index(current, &p2)
                        child = self.create_child(current, index)

                        # store old particle here
                        child.p.mass = p2.mass
                        for j in range(self.dim):
                            child.p.x[j] = p2.x[j]

                        # try again to insert original particle

        self._update_moments(0, -1)
        if self.parallel:
            self._export_import_remote_nodes()

    cdef void _update_moments(self, int current, int sibling):
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

    cdef void _export_import_remote_nodes(self):
        cdef int i, j
        cdef Node* node
        cdef np.float64_t* comx[3]

        cdef LongArray proc   = self.remote_nodes.get_carray('proc')
        cdef LongArray maps   = self.remote_nodes.get_carray('map')
        cdef DoubleArray mass = self.remote_nodes.get_carray('mass')

        self.remote_nodes.pointer_groups(comx, self.remote_nodes.named_groups['com'])

        # collect moments belonging to current processor
        for i in range(self.remote_nodes.length):
            if proc.data[i] == self.pid:

                # copy moment information
                node = self.nodes.array[maps.data[i]]
                for j in range(self.dim):
                    comx[j][i] = node.p.x[j]
                mass.data[i] = node.p.m

        # export local node info and import remote node
        for field in self.remote_nodes.named_groups['moments']:
            comm.Allgatherv(MPI.IN_PLACE,
                    [self.remote_nodes[field], self.node_counts, self.node_disp, MPI.DOUBLE])

        # update moments
        self._update_remote_moments(0)

    cdef void _update_remote_moments(self, int current):
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

                    # update node moments
                    child = &self.nodes.node_array[node.children[i]]
                    if child.dependant:
                        self._update_remote_moments(node.children[i])

                    node.p.mass += child.p.mass
                    for j in range(self.dim):
                        node.p.x[j] += child.p.mass*child.p.x[j]

            # guard against if node has pseudo particles
            if(node.p.mass):
                for j in range(self.dim):
                    node.p.x[j] /= node.p.mass

#    def walk(self, Interaction interact, CarrayContainer pc):
#        self._walk(interact, pc)
#
#    cdef void _walk(self, Interaction interaction, CarrayContainer pc):
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
#                    if(interaction.splitter.split(node)):
#                        # node opened travel down
#                        index = node.first_child
#                    else:
#                        # interaction: node-particle
#                        interaction.interact(node)
#                        index = node.next_sibling
#                else:
#                    # interaction: particle-particle
#                    interaction.interact(node)
#                    index = node.next_sibling


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


#    '''
#    1. process local particles
#    2. store particle if flagged
#    3. organize particles for export
#    4. do you really want to process by processor chunk or as a whole
#    5. if by chunk then this is easier? just compute for each particle
#       and send back?
#
#    1. sort
#    2. count data on eac processor
#    3. allocate buffer
#    4. copy field to buffers
#    5. for each processor with data:
#        mpi buffer sendjjjjjjjjjj
#    '''
#    cdef void add_data(int index, int pid):
#
#        if self.ind_buffer.length + 1 > self.buffer_size:
#            self.communicate()
#            self.flush_buffer()
#        else:
#            self.ind_buffer.append(index)
#            self.pro_buffer.append(pid)
#
#    cdef void _walk(self, Interaction interaction, CarrayContainer pc):
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
#        # communicator 
#        comm.setup_callback_func(self._parallel_walk)
#        comm.setup_datatype(pc, pc.named_groups['tree-gravity'])
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
#                comm.flush_send()
#
#                node = &self.nodes.node_array[index]
#                if(node.leaf != LEAF):
#
#                    if(interaction.splitter.split(node)):
#                        # node opened travel down
#                        index = node.first_child
#                    else:
#                        # interaction: node-particle
#                        interaction.interact(node)
#                        index = node.next_sibling
#                else:
#                    # node on other processor
#                    if node.pid != pid:
#
#                        if self.buffer_pid.size + 1 < comm.buffer_size:
#                            self.buffer_pid.append(node.pid)
#                            self.buffer_id.append(i)
#
#                        else:
#
#                            comm.setup_outgoing_datatype(pc, pc.named_groups['tree-gravity'])
#                            comm.communicate_data()
#
#                            self.parallel_walk(comm.buffer_out)
#
#                            comm.setup_outgoing_datatype(pc, pc.named_groups['tree-gravity'])
#                            comm.communicate_data()
#
#                            comm.flush_buffer()
#
#                            self.buffer_pid.append(node.pid)
#                            self.buffer_id.append(i)
#
#                    # interaction: particle-particle
#                    interaction.interact(node)
#                    index = node.next_sibling
