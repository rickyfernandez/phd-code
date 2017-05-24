import numpy as np
cimport numpy as np
cimport libc.stdlib as stdlib

from mpi4py import MPI
from libc.math cimport sqrt

from ..utils.exchange_particles import exchange_particles

from .splitter cimport Splitter, BarnesHut
from .interaction cimport GravityAcceleration
from ..utils.particle_tags import ParticleTAGS
from ..load_balance.tree cimport TreeMemoryPool as Pool
from ..utils.carray cimport DoubleArray, IntArray, LongArray, LongLongArray

cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost

cdef int proc_compare(const void *a, const void *b):
    if( (<PairId*>a).proc < (<PairId*>b).proc ):
        return -1
    if( (<PairId*>a).proc > (<PairId*>b).proc ):
        return 1
    return 0

cdef class GravityTree:
    """
    Solves gravity by barnes and hut algorithm in serial or
    parallel. The algorithm heavily depends on the LoadBalance
    class if run in parallel. The algorithm works in 2d or 3d.
    """
    def __init__(self, str split_type='barnes-hut',  double barnes_angle=0.3,
            int calculate_potential=0, int parallel=0, int max_buffer_size=256):

        #self.domain = domain
        #self.load_bal = load_bal
        #self.pc = pc
        self.split_type = split_type
        self.barnes_angle = barnes_angle
        self.calc_potential = calculate_potential

        self.parallel = parallel
        self.max_buffer_size = max_buffer_size

    def _initialize(self):
        '''
        Initialize variables for the gravity tree. Tree pool and
        remote nodes are allocated as well dimension of the tree.
        '''
        cdef str axis
        cdef Splitter splitter
        cdef dict state_vars = {}
        cdef dict named_groups = {}

        self.dim = self.domain.dim
        self.number_nodes = 2**self.dim
        self.nodes = GravityPool(1000)

        if self.split_type == 'barnes-hut':
            splitter = BarnesHut(self.dim, self.barnes_angle)
        else:
            raise RuntimeError("Unrecognized splitter in gravity")

        self.export_interaction = GravityAcceleration(self.pc,
                self.domain, splitter, 1, self.calc_potential)

        self.rank = self.size = 0

        if self.parallel:

            self.rank = self.load_bal.rank
            self.size = self.load_bal.size

            self.send_cnts = np.zeros(self.size, dtype=np.int32)
            self.send_disp = np.zeros(self.size, dtype=np.int32)

            self.recv_cnts = np.zeros(self.size, dtype=np.int32)
            self.recv_disp = np.zeros(self.size, dtype=np.int32)

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

            # particle id and send processors buffers
            self.buffer_ids = <PairId*> stdlib.malloc(
                    self.max_buffer_size*sizeof(PairId))
            if self.buffer_ids == NULL:
                raise MemoryError("Insufficient memory in id buffer")
            self.buffer_size = 0

            self.indices = LongArray(n=self.max_buffer_size)
            self.flag_pid = np.zeros(self.size, dtype=np.int32)

            # particle buffers for parallel tree walk
            self.buffer_import = CarrayContainer(0)
            self.buffer_export = CarrayContainer(0)

            # add fields that will be communicated
            for field in self.pc.named_groups['gravity']:
                self.buffer_export.register_property(0, field,
                        self.export_interaction.fields[field])
                self.buffer_import.register_property(0, field,
                        self.export_interaction.fields[field])

            # add name groups as well
            self.buffer_export.named_groups['acceleration'] =\
                    list(self.pc.named_groups['acceleration'])
            self.buffer_export.named_groups['position'] =\
                    list(self.pc.named_groups['position'])

            self.buffer_import.named_groups['acceleration'] =\
                    list(self.pc.named_groups['acceleration'])
            self.buffer_import.named_groups['position'] =\
                    list(self.pc.named_groups['position'])

            self.import_interaction = GravityAcceleration(self.pc,
                    self.domain, splitter, 0, self.calc_potential)

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
        cdef Node* node = &self.nodes.array[parent_index]
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
        parent = &self.nodes.array[parent_index]
        parent.group.children[child_index] = self.nodes.used - 1

        # parent no longer leaf
        parent.flags &= ~LEAF

        for i in range(self.number_nodes):
            child.group.children[i] = NOT_EXIST

        child.flags = LEAF
        width = .5*parent.width
        child.width = width

        for i in range(self.dim):
            # create center coords for child
            if((child_index >> i) & 1):
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

        parent = &self.nodes.array[parent_index]
        width = .5*parent.width                            # box width of children
        parent.flags &= ~LEAF                              # parent no longer leaf

        # loop over each child and pass parent information
        for i in range(self.number_nodes):

            # store child index in node array
            parent.group.children[i] = start_index + i
            child = &self.nodes.array[start_index + i]

            child.flags = LEAF
            child.width = width

            # set children of children to null
            for k in range(self.number_nodes):
                child.group.children[k] = NOT_EXIST

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
        cdef int i, pid
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

        self.node_index_to_array.clear()

        # remote nodes are in load balance order, hilbert and processor, this
        # allows for easy communication. Transfer processor id
        for i in range(self.size):
            self.send_cnts[i] = 0

        for i in range(leaf_pid.length):

            pid = leaf_pid.data[i]
            proc.data[i] = pid
            self.send_cnts[pid] += 1

            # reverse mapping for leafs
            self.node_index_to_array[maps.data[i]] = i

            if(pid != self.rank):
                node = &self.nodes.array[maps.data[i]]
                node.flags |= (SKIP_BRANCH|TOP_TREE_LEAF_REMOTE)

        self.send_disp[0] = 0
        for i in range(1, self.size):
            self.send_disp[i] = self.send_cnts[i-1] + self.send_disp[i-1]

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
        cdef Node* parent = &self.nodes.array[node_index]

        # label node in top tree
        parent.flags |= TOP_TREE

        if load_parent.children_start == -1: # leaf stop
            parent.flags |= TOP_TREE_LEAF
            node_map[load_parent.array_index] = node_index
        else: # non leaf copy

            # create children in z-order
            self.create_children(node_index)

            # create children could of realloc
            parent = &self.nodes.array[node_index]

            # travel down to children
            for i in range(self.number_nodes):

                # grab next child in z-order
                index = load_parent.zorder_to_hilbert[i]
                self._create_top_tree(
                        parent.group.children[i], load_parent + load_parent.children_start + index,
                        node_map)

    cdef inline int _leaf_index_toptree(self, np.int64_t key):
        """
        Find index of local tree which coincides with given key
        inside leaf in top tree.
        """
        cdef LoadNode* load_node
        cdef LongArray maps = self.remote_nodes.get_carray('map')

        load_node = self.load_bal.tree.find_leaf(key)
        return maps.data[load_node.array_index]

    cdef void create_root(self):
        """
        Reset tree if needed and allocate one node for
        the root and transfer domain information to the
        root.
        """
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
            root.group.children[k] = NOT_EXIST

    def _build_tree(self):
        """
        Build local gravity tree by inserting real particles.
        This method is non-recursive and only adds real particles.
        Note, leaf nodes may have a particle. The distinction is for
        parallel tree builds because the top tree will have leafs
        without any particles.
        """
        cdef IntArray tags = self.pc.get_carray('tag')
        cdef DoubleArray mass = self.pc.get_carray('mass')
        cdef LongLongArray keys = self.pc.get_carray('key')

        cdef double width
        cdef int index, current
        cdef Node *node, *child

        cdef int i, j, k
        cdef double xi[3], xj[3]

        # pointer to particle position and mass
        self.pc.pointer_groups(self.x, self.pc.named_groups['position'])
        self.m = mass.get_data_ptr()

        self.create_root()

        if self.parallel:
            self._build_top_tree()

        # add real particles to tree
        for i in range(self.pc.get_number_of_items()):
            if tags.data[i] == Real:

                for k in range(self.dim):
                    xi[k] = self.x[k][i]

                if self.parallel: # start at top tree leaf
                    current = self._leaf_index_toptree(keys.data[i])
                else: # start at root
                    current = ROOT

                while True:
                    node = &self.nodes.array[current]
                    if (node.flags & LEAF):
                        if (node.flags & HAS_PARTICLE):

                            # leaf has particle already
                            j = node.group.data.pid
                            for k in range(self.dim):
                                xj[k] = self.x[k][j]

                            # reset children to null due to union
                            for k in range(self.number_nodes):
                                node.group.children[k] = NOT_EXIST

                            # node becomes internal node
                            node.flags &= ~(LEAF|HAS_PARTICLE)

                            # create child to store leaf particle
                            index = self.get_index(current, xj)
                            child = self.create_child(current, index)

                            # store leaf particle here
                            child.flags |= (LEAF|HAS_PARTICLE)
                            child.group.data.pid = j

                            # try to insert original particle again

                        else:
                            node.flags |= HAS_PARTICLE # store particle here
                            node.group.data.pid = i    # overwrites in union
                            break # particle done

                    else:
                        # find child to store particle
                        index = self.get_index(current, xi)

                        # if child does not exist create child
                        # and store particle
                        if node.group.children[index] == NOT_EXIST:
                            child = self.create_child(current, index)
                            child.flags |= (LEAF|HAS_PARTICLE)
                            child.group.data.pid = i
                            break # particle done

                        else: # internal node, travel down
                            current = node.group.children[index]

        # calculate node moments
        self._update_moments(ROOT, ROOT_SIBLING)

        if self.parallel:
            # export top tree leaf moments and
            # recalculate node moments
            self._export_import_remote_nodes()

        self.m = NULL
        for k in range(self.dim):
            self.x[k] = NULL

    cdef void _update_moments(self, int current, int sibling):
        """
        Recursively update moments of each local node. As a by
        product we collect the first child and sibling of each
        node, which allows for efficient tree walking.
        """
        cdef Node *node, *child
        cdef double mass, com[3]
        cdef int i, j, k, sib, pid, skip

        skip = SKIP_BRANCH

        # due to union we first save moments
        mass = 0.
        for i in range(self.dim):
            com[i] = 0.
        node = &self.nodes.array[current]

        # for non leafs use children
        if((node.flags & LEAF) != LEAF):
            for i in range(self.number_nodes):
                if(node.group.children[i] != NOT_EXIST):

                    # find sibling of child 
                    j = i + 1
                    while(j < self.number_nodes and\
                            node.group.children[j] == NOT_EXIST):
                        j += 1

                    if(j < self.number_nodes):
                        sib = node.group.children[j]
                    else:
                        sib = sibling

                    self._update_moments(node.group.children[i], sib)
                    child = &self.nodes.array[node.group.children[i]]

                    # for parallel flag branches to skip during walk
                    skip &= (child.flags & SKIP_BRANCH)

                    mass += child.group.data.mass
                    for k in range(self.dim):
                        com[k] += child.group.data.mass*\
                                child.group.data.com[k]

            # find first child of node
            j = 0
            while(node.group.children[j] == NOT_EXIST):
                j += 1

            node.flags |= (skip & SKIP_BRANCH)

            # no longer need children array in union
            node.group.data.first_child = node.group.children[j]
            node.group.data.next_sibling = sibling
            node.group.data.mass = mass

            if(mass):
                for k in range(self.dim):
                    node.group.data.com[k] = com[k]/mass
            else:
                for k in range(self.dim):
                    node.group.data.com[k] = 0.
        else:

            node.group.data.first_child  = NOT_EXIST
            node.group.data.next_sibling = sibling

            # remote leafs may not have particles
            if(node.flags & HAS_PARTICLE):
                pid = node.group.data.pid

                # copy particle information
                node.group.data.mass = self.m[pid]
                for k in range(self.dim):
                    node.group.data.com[k] = self.x[k][pid]
            else:
                node.group.data.mass = 0.
                for k in range(self.dim):
                    node.group.data.com[k] = 0.

    cdef void _update_remote_moments(self, int current):
        """
        Recursively update moments of each local node. As a by
        product we collect the first child and sibling of each
        node. This in turn allows for efficient tree walking.
        """
        cdef int k, ind, sib
        cdef Node *node, *child
        cdef double mass, com[3]

        node = &self.nodes.array[current]

        # check if node is not a top tree leaf
        if((node.flags & TOP_TREE_LEAF) != TOP_TREE_LEAF):

            mass = 0.
            for k in range(self.dim):
                com[k] = 0.

            # sum moments from each child
            ind = node.group.data.first_child
            sib = node.group.data.next_sibling
            while(ind != sib):

                # update node moments
                child = &self.nodes.array[ind]
                self._update_remote_moments(ind)
                mass += child.group.data.mass
                for k in range(self.dim):
                    com[k] += child.group.data.mass*\
                            child.group.data.com[k]

                ind = child.group.data.next_sibling

            if(mass):
                for k in range(self.dim):
                    com[k] /= mass

            node.group.data.mass = mass
            for k in range(self.dim):
                node.group.data.com[k] = com[k]

    cdef void _export_import_remote_nodes(self):
        cdef int i, j
        cdef Node *node
        cdef np.float64_t* comx[3]

        cdef LongArray proc   = self.remote_nodes.get_carray('proc')
        cdef LongArray maps   = self.remote_nodes.get_carray('map')
        cdef DoubleArray mass = self.remote_nodes.get_carray('mass')

        self.remote_nodes.pointer_groups(comx,
                self.remote_nodes.named_groups['com'])

        # collect moments belonging to current processor
        for i in range(self.remote_nodes.get_number_of_items()):
            if proc.data[i] == self.rank:

                node = &self.nodes.array[maps.data[i]]
                for j in range(self.dim):
                    comx[j][i] = node.group.data.com[j]
                mass.data[i] = node.group.data.mass

        # export local node info and import remote node
        for field in self.remote_nodes.named_groups['moments']:
            self.load_bal.comm.Allgatherv(MPI.IN_PLACE,
                    [self.remote_nodes[field], self.send_cnts,
                        self.send_disp, MPI.DOUBLE])

        # copy remote nodes to tree
        for i in range(self.remote_nodes.get_number_of_items()):
            if proc.data[i] != self.rank:

                node = &self.nodes.array[maps.data[i]]
                for j in range(self.dim):
                    node.group.data.com[j] = comx[j][i]
                node.group.data.mass = mass.data[i]

        # recalculate moments
        self._update_remote_moments(ROOT)

    def walk(self):
        """
        Walk the tree calculating accerlerations.
        """
        self.export_interaction.initialize_particles(self.pc)
        if self.parallel:
            self._parallel_walk(self.export_interaction)
        else:
            self._serial_walk(self.export_interaction)

    cdef void _serial_walk(self, Interaction interaction):
        """
        Walk the tree calculating interactions. Interactions can be any
        calculation between particles.

        Parameters
        ----------
        interaction : Interaction
            Computation for particle and node
        """
        cdef int index
        cdef Node *node

        # loop through each real praticle
        while(interaction.process_particle()):
            index = ROOT
            while(index != ROOT_SIBLING):
                node = &self.nodes.array[index]

                if(node.flags & LEAF):
                    # calculate particle particle interaction
                    interaction.interact(node)
                    index = node.group.data.next_sibling
                else:
                    if(interaction.splitter.split(node)):
                        # node opened travel down
                        index = node.group.data.first_child
                    else:
                        # calculate node particle interaction
                        interaction.interact(node)
                        index = node.group.data.next_sibling

    cdef void _import_walk(self, Interaction interaction):
        """
        Walk tree calculating interactions for particle that are
        imported to this process.
        """
        cdef int index
        cdef Node *node

        # loop through each export praticle
        while(interaction.process_particle()):
            index = ROOT
            while(index != ROOT_SIBLING):
                node = &self.nodes.array[index]

                if(node.flags & LEAF):
                    if(node.flags & TOP_TREE_LEAF_REMOTE):
                        # skip remote leaf
                        index = node.group.data.next_sibling

                    else: # calculate particle particle interaction
                        interaction.interact(node)
                        index = node.group.data.next_sibling

                else: # node is not leaf
                    if(node.flags & SKIP_BRANCH):
                        # we can skip branch if node only depends
                        # on remote nodes
                        index = node.group.data.next_sibling

                    # check if node can be opened
                    elif(interaction.splitter.split(node)):
                        # travel down node
                        index = node.group.data.first_child

                    else: # node not opened particle node interaction
                        if(node.flags & TOP_TREE):
                            # skip top tree nodes
                            index = node.group.data.next_sibling
                        else:
                            # calculate node particle interaction
                            interaction.interact(node)
                            index = node.group.data.next_sibling

    cdef void _export_walk(self, Interaction interaction):
        """
        Walk tree calculating interactions for particle on this
        process. Particle are also flagged for export.
        """
        cdef Node *node
        cdef int index, i, node_pid

        cdef LongArray pid = self.remote_nodes.get_carray("proc")

        # loop through each real praticle
        while(interaction.process_particle()):

            # clear out export to processor flag
            for i in range(self.size):
                self.flag_pid[i] = 0

            # start at root or next node from previous walk
            index = interaction.start_node_index()
            while(index != -1):

                node = &self.nodes.array[index]
                if(node.flags & LEAF):
                    if(node.flags & TOP_TREE_LEAF_REMOTE):
                        if(interaction.splitter.split(node)):

                            # node opend check if particle alreay flagged
                            node_pid = pid.data[self.node_index_to_array[index]]
                            if self.flag_pid[node_pid]:
                                index = node.group.data.next_sibling
                            else:

                                # particle exported to pid
                                self.flag_pid[node_pid] = 1

                                # node opened flag for export
                                self.buffer_ids[self.buffer_size].index =\
                                        interaction.current
                                self.buffer_ids[self.buffer_size].proc =\
                                        node_pid

                                self.buffer_size += 1

                                # check if buffer is full, halt walk if true
                                if self.buffer_size == self.max_buffer_size:
                                    # save node to continue walk
                                    interaction.particle_not_finished(
                                            node.group.data.next_sibling)
                                    return # break out of walk
                                else:
                                    index = node.group.data.next_sibling

                        else: # node not opened particle node interaction
                            interaction.interact(node)
                            index = node.group.data.next_sibling

                    else: # particle particle interaction
                        interaction.interact(node)
                        index = node.group.data.next_sibling

                else: # check if node can be opened 
                    if(interaction.splitter.split(node)):
                        # travel down node
                        index = node.group.data.first_child

                    else: # node not opened particle node interaction
                        interaction.interact(node)
                        index = node.group.data.next_sibling

            interaction.particle_finished()
        interaction.done_processing()

    cdef void _parallel_walk(self, Interaction interaction):
        cdef int i
        cdef long num_import
        cdef int local_done, global_done
        cdef np.ndarray loc_done, glb_done

        loc_done = np.zeros(1, dtype=np.int32)
        glb_done = np.zeros(1, dtype=np.int32)

        # clear out buffers
        self.buffer_size = 0

        # setup local particles for walk
        self.export_interaction.initialize_particles(self.pc)
        while True:

            # reset buffers
            self.buffer_size = 0
            self.indices.resize(self.buffer_size)
            self.buffer_import.resize(self.buffer_size)
            self.buffer_export.resize(self.buffer_size)

            # reset import/export counts
            for i in range(self.size):
                self.send_cnts[i] = 0
                self.recv_cnts[i] = 0

            # perform walk while flagging particles for export
            # once the buffer is full the walk will hault
            self._export_walk(self.export_interaction)
            if self.buffer_size:

                # put particles in process order
                qsort(<void*> self.buffer_ids, <size_t> self.buffer_size,
                        sizeof(PairId), proc_compare)

                # copy particle indices in process order and count
                # the number number particles export per processor
                self.indices.resize(self.buffer_size)
                for i in range(self.buffer_size):
                    self.indices.data[i] = self.buffer_ids[i].index
                    self.send_cnts[self.buffer_ids[i].proc] += 1

                # copy flagged particles into buffer
                self.buffer_export.resize(self.buffer_size)
                self.buffer_export.copy(self.pc, self.indices,
                        self.pc.named_groups['gravity-walk-export'])

            # send number of exports to all processors
            self.load_bal.comm.Alltoall([self.send_cnts, MPI.INT],
                    [self.recv_cnts, MPI.INT])

            # how many remote particles are incoming
            num_import = 0
            for i in range(self.size):
                num_import += self.recv_cnts[i]

            # create displacement arrays 
            self.send_disp[0] = self.recv_disp[0] = 0
            for i in range(1, self.size):
                self.send_disp[i] = self.send_cnts[i-1] + self.send_disp[i-1]
                self.recv_disp[i] = self.recv_cnts[i-1] + self.recv_disp[i-1]

            # resize to fit number of particle incoming
            self.buffer_import.resize(num_import)

            # send our particles / recieve particles 
            exchange_particles(self.buffer_import, self.buffer_export,
                    self.send_cnts, self.recv_cnts,
                    0, self.load_bal.comm,
                    self.pc.named_groups['gravity-walk-export'],
                    self.send_disp, self.recv_disp)

            # walk remote particles
            self.import_interaction.initialize_particles(self.buffer_import, 0)
            self._import_walk(self.import_interaction)

            # recieve back our paritcles / send back particles
            exchange_particles(self.buffer_export, self.buffer_import,
                    self.recv_cnts, self.send_cnts,
                    0, self.load_bal.comm,
                    self.pc.named_groups['gravity-walk-import'],
                    self.recv_disp, self.send_disp)

            # copy back our data
            self.pc.add(self.buffer_export, self.indices,
                    self.pc.named_groups['gravity-walk-import'])

            # let all processors know if walk is complete 
            glb_done[0] = 0
            loc_done[0] = self.export_interaction.done_processing()
            self.load_bal.comm.Allreduce([loc_done, MPI.INT], [glb_done, MPI.INT], op=MPI.SUM)

            # if all processors tree walks are done exit
            if glb_done[0] == self.size:
                break

    # -- delete later --
    def dump_root_node(self):
        cdef Node* root
        cdef np.float64_t mass, width

        root = &self.nodes.array[0]
        width = root.width
        pos = [
            root.center[0],
            root.center[1],
            root.center[2]]

        mass = root.group.data.mass
        com = [
            root.group.data.com[0],
            root.group.data.com[1],
            root.group.data.com[2]]

        return pos, width, mass, com

    # -- delete later --
    def dump_data(self):
        cdef list data_list = []
        cdef Node* node

        cdef int i
        for i in range(self.nodes.used):
            node = &self.nodes.array[i]
            if (node.flags & LEAF):
                data_list.append([
                    node.center[0],
                    node.center[1],
                    node.center[2],
                    node.group.data.mass,
                    node.group.data.com[0],
                    node.group.data.com[1],
                    node.width])

        return data_list

    # -- delete later --
    def dump_all_data(self):
        cdef list data_list = []
        cdef Node* node

        cdef int i
        for i in range(self.nodes.used):
            node = &self.nodes.array[i]
            data_list.append([
                node.center[0],
                node.center[1],
                node.center[2],
                node.group.data.mass,
                node.group.data.com[0],
                node.group.data.com[1],
                (node.flags) & LEAF,
                node.width])

        return data_list

    # -- delete later --
    def dump_remote(self):
        cdef int i, j
        cdef Node* node
        cdef list data_list = []
        cdef LongArray maps = self.remote_nodes.get_carray('map')
        cdef LongArray proc = self.remote_nodes.get_carray('proc')

        for i in range(self.remote_nodes.get_number_of_items()):
            j = maps.data[i]
            node = &self.nodes.array[j]
            data_list.append([
                node.center[0],
                node.center[1],
                node.center[2],
                node.group.data.mass,
                node.group.data.com[0],
                node.group.data.com[1],
                proc.data[i],
                node.width])

        return data_list

    def __dealloc__(self):
        """
        Deallocate buffers in gravity
        """
        stdlib.free(self.buffer_ids)
