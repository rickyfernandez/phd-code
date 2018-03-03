import phd
import numpy as np
cimport numpy as np
cimport libc.stdlib as stdlib

from mpi4py import MPI
from libc.math cimport sqrt

from ..utils.particle_tags import ParticleTAGS
from ..utils.exchange_particles import exchange_particles

from .splitter cimport Splitter, BarnesHut
from .interaction cimport GravityAcceleration
from ..load_balance.tree cimport TreeMemoryPool as Pool
from ..utils.carray cimport DoubleArray, IntArray, LongArray, LongLongArray

cdef int Real = ParticleTAGS.Real
cdef int Ghost = ParticleTAGS.Ghost

cdef int proc_compare(const void *a, const void *b):
    """
    Comparison function for sorting PairId struct
    in processor order.
    """
    if( (<PairId*>a).proc < (<PairId*>b).proc ):
        return -1
    if( (<PairId*>a).proc > (<PairId*>b).proc ):
        return 1
    return 0

cdef class GravityTree:
    """Solves gravity by barnes and hut algorithm in serial
    or parallel. The algorithm heavily depends on LoadBalance
    class if run in parallel. The algorithm works in 2d or 3d.
    """
    def __init__(self, str split_type="barnes-hut",  double barnes_angle=0.3,
            double smoothing_length = 1.0E-5, int calculate_potential=0,
            int max_buffer_size=256):

        self.split_type = split_type
        self.barnes_angle = barnes_angle
        self.calculate_potential = calculate_potential

        self.load_balance = None
        self.domain_manager = None
        self.max_buffer_size = max_buffer_size

        # criteria to open nodes
        if self.split_type == "barnes-hut":
            self.export_splitter = BarnesHut(self.barnes_angle)

            if phd._in_parallel:
                self.import_splitter = BarnesHut(self.barnes_angle)
        else:
            raise RuntimeError("Unrecognized splitter in gravity")

        # gravity force caculator
        self.export_interaction = GravityAcceleration(
                calculate_potential, smoothing_length)

        if phd._in_parallel:

            self.import_interaction = GravityAcceleration(
                    calculate_potential, smoothing_length)

            self.indices = LongArray(n=self.max_buffer_size)

            # particle buffers for parallel tree walk
            self.buffer_import = CarrayContainer(0)
            self.buffer_export = CarrayContainer(0)

    def register_fields(self, CarrayContainer particles):
        """Register gravity fields into the particle container (i.e.
        accleration, potential, mpi groups etc.).

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef str field
        self.export_interaction.register_fields(particles)

        if phd._in_parallel:

            # hack because export already registered
            self.import_interaction.dim = self.export_interaction.dim
            self.import_interaction.particle_fields_registered = True

            # add fields that will be communicated
            for field in particles.carray_named_groups["gravity"]:

                self.buffer_export.register_carray(0, field,
                        particles.carray_dtypes[field])

                self.buffer_import.register_carray(0, field,
                        particles.carray_dtypes[field])

            # add name groups as well
            self.buffer_export.carray_named_groups["acceleration"] =\
                    list(particles.carray_named_groups["acceleration"])
            self.buffer_export.carray_named_groups["position"] =\
                    list(particles.carray_named_groups["position"])

            self.buffer_import.carray_named_groups["acceleration"] =\
                    list(particles.carray_named_groups["acceleration"])
            self.buffer_import.carray_named_groups["position"] =\
                    list(particles.carray_named_groups["position"])

    def add_fields(self, CarrayContainer particles):
        """Setup containers for toptree nodes.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef str axis
        cdef dict toptree_carray_to_register = {}
        cdef dict toptree_carray_named_groups = {}

        self.dim = len(particles.carray_named_groups["position"])

        if phd._in_parallel:

            toptree_carray_to_register["map"]  = "long"
            toptree_carray_to_register["proc"] = "long"
            toptree_carray_to_register["mass"] = "double"

            toptree_carray_named_groups["com"] = []
            for axis in "xyz"[:self.dim]:
                toptree_carray_to_register["com-"+axis] = "double"
                toptree_carray_named_groups["com"].append("com-"+axis)

            toptree_carray_named_groups["moments"] = ["mass"] +\
                    toptree_carray_named_groups["com"]

            self.toptree_carray_to_register = toptree_carray_to_register
            self.toptree_carray_named_groups = toptree_carray_named_groups

    def set_domain_manager(self, DomainManager domain_manager):
        """Set domain manager."""
        self.domain_manager = domain_manager

    def initialize(self):
        """Initialize variables for the gravity tree. Tree pool and
        toptree nodes are allocated as well dimension of the tree.
        """
        cdef str axis

        if not self.domain_manager:
            raise RuntimeError("ERROR: DomainManager not set")

        self.export_splitter.set_dim(self.dim)
        self.export_interaction.set_splitter(self.export_splitter)
        self.export_interaction.initialize()

        if phd._in_parallel:

            self.import_splitter.set_dim(self.dim)
            self.import_interaction.set_splitter(self.import_splitter)
            self.import_interaction.initialize()

        self.number_nodes = 2**self.dim
        self.nodes = GravityPool(10000)

        if phd._in_parallel:

            self.load_balance = self.domain_manager.load_balance

            # export processor counts and displacements
            self.send_cnts = np.zeros(phd._size, dtype=np.int32)
            self.send_disp = np.zeros(phd._size, dtype=np.int32)

            # import processor counts and displacements
            self.recv_cnts = np.zeros(phd._size, dtype=np.int32)
            self.recv_disp = np.zeros(phd._size, dtype=np.int32)

            # container of nodes common to all processors
            self.toptree_leafs = CarrayContainer(
                    carrays_to_register=self.toptree_carray_to_register)

            self.toptree_leafs.carray_named_groups =\
                    self.toptree_carray_named_groups

            # particle id and send processors buffers
            self.buffer_ids = <PairId*> stdlib.malloc(
                    self.max_buffer_size*sizeof(PairId))
            if self.buffer_ids == NULL:
                raise MemoryError("ERROR: Insufficient memory in id buffer")
            self.buffer_size = 0

    cdef inline int _get_index(self, int parent_index, np.float64_t x[3]):
        """Return index of child from parent node with node_index.
        Children are laid out in z-order.

        Parameters
        ----------
        node_index : int
            Index of node that you want to find child of.

        x : np.float64_t[3]
            Particle coordinates to find child.

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

    cdef inline Node* _create_child(self, int parent_index, int child_index):
        """Create child node given parent index and child index. Note
        parent_index refers to memory pool and child_index refers to
        [0,3] in 2d or [0,7] for children array in parent.

        Parameters
        ----------
        parent_index : int
            Index of parent in pool.

        child_index : int
            Index of child relative to parent.children array.

        Returns
        -------
        child : Node*
            child pointer

        """
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

    cdef inline void _create_children(self, int parent_index):
        """Given a parent node, subdivide node into (4-2d, 8-3d)
        children.

        Parameters
        ----------
        parent_index : int
            Index of parent in pool.

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

    cdef void _build_toptree(self):
        """Copy the load balance tree. The tree is the starting point
        to add particles since this tree is common to all processors.
        Note the load balance tree is in hilbert order.
        """
        cdef int i, pid
        cdef np.int32_t *node_ind

        cdef Node *node
        cdef LoadNode *load_root = self.load_balance.tree.root

        cdef Pool pool = self.load_balance.tree.mem_pool

        cdef LongArray leaf_pid = self.load_balance.leaf_pid
        cdef LongArray maps = self.toptree_leafs.get_carray("map")
        cdef LongArray proc = self.toptree_leafs.get_carray("proc")

        # resize memory pool to hold tree - this only allocates available
        # memory it does not create nodes
        self.nodes.resize(pool.number_nodes())

        # resize container to hold load leaf data 
        self.toptree_leafs.resize(pool.number_leaves())

        # copy global top tree in z-order, collect load leaf index for mapping 
        self._create_toptree(ROOT, load_root, maps.get_data_ptr())

        # reset top tree leaf to toptree container map
        self.toptree_leaf_map.clear()

        # top tree leafs are in load balance order, hilbert and processor,
        # this allows for easy communication.
        for i in range(phd._size):
            self.send_cnts[i] = 0

        # loop over load leafs
        for i in range(leaf_pid.length):

            pid = leaf_pid.data[i]     # processor that owns leaf
            proc.data[i] = pid         # store processor info

            # bin leafs to processor - our processor is bined
            # because we comunicate IN_PLACE in mpi
            self.send_cnts[pid] += 1

            # mapping toptree leaf index -> toptree leaf container 
            self.toptree_leaf_map[maps.data[i]] = i

            # flag leafs that don't belong to this processor
            if(pid != phd._rank):
                node = &self.nodes.array[maps.data[i]]
                node.flags |= (SKIP_BRANCH|TOP_TREE_LEAF_REMOTE)

        self.send_disp[0] = 0
        for i in range(1, phd._size):
            self.send_disp[i] = self.send_cnts[i-1] + self.send_disp[i-1]

    cdef void _create_toptree(self, int node_index, LoadNode* load_parent,
            np.int32_t* node_map):
        """Copys the load balance tree. The tree is the starting point
        to add particles since this tree is common to all processors.
        Note the load balance tree is in hilbert order, so care is
        taken to put the gravity tree in z-order. Note the leafs of the
        top tree are the objects used for the load balance. The leafs
        are stored in toptree_leafs container and are in hilbert and
        processor order. The map array is used to map from toptree_leafs
        to nodes in the gravity tree. All nodes will be labeled to
        belong to the top tree.

        Parameters
        ----------
        node_index : int
            Index of node in gravity tree.

        load_parent : LoadNode*
            Node pointer to load balance tree.

        node_map : np.int32_t*
            Array to map container index to toptree leaf.

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
            self._create_children(node_index)

            # create children could of realloc
            parent = &self.nodes.array[node_index]

            # travel down to children
            for i in range(self.number_nodes):

                # grab next child in z-order
                index = load_parent.zorder_to_hilbert[i]
                self._create_toptree(
                        parent.group.children[i], load_parent + load_parent.children_start + index,
                        node_map)

    cdef inline int _leaf_index_toptree(self, np.int64_t key):
        """Find index of local tree which coincides with given key
        inside leaf in top tree.

        Parameters
        ----------
        key : np.int64_t
            Hilbert key of node

        """
        cdef LoadNode* load_node
        cdef LongArray maps = self.toptree_leafs.get_carray('map')

        load_node = self.load_balance.tree.find_leaf(key)
        return maps.data[load_node.array_index]

    cdef inline void _create_root(self):
        """Reset tree if needed and allocate one node for
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
        root.width = self.domain_manager.max_length
        for k in range(self.dim):
            root.center[k] = .5*\
                    (self.domain_manager.bounds[1][k] - self.domain_manager.bounds[0][k])

        # set root children to null
        for k in range(self.number_nodes):
            root.group.children[k] = NOT_EXIST

    def _build_tree(self, CarrayContainer particles):
        """Build local gravity tree by inserting real particles.
        This method is non-recursive and only adds real particles.
        Note, leaf nodes may have a particle. The distinction is for
        parallel tree builds because the top tree will have leafs
        without any particles.
        """
        cdef IntArray tags = particles.get_carray("tag")
        cdef DoubleArray mass = particles.get_carray("mass")
        cdef LongLongArray keys = particles.get_carray("key")

        cdef double width
        cdef int index, current
        cdef Node *node, *child

        cdef int i, j, k
        cdef double xi[3], xj[3]

        # pointer to particle position and mass
        particles.pointer_groups(self.x, particles.carray_named_groups["position"])
        self.m = mass.get_data_ptr()

        self._create_root()

        if phd._in_parallel:
            self._build_toptree()

        # add real particles to tree
        for i in range(particles.get_carray_size()):
            if tags.data[i] == Real:

                for k in range(self.dim):
                    xi[k] = self.x[k][i]

                if phd._in_parallel: # start at top tree leaf
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
                            index = self._get_index(current, xj)
                            child = self._create_child(current, index)

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
                        index = self._get_index(current, xi)

                        # if child does not exist create child
                        # and store particle
                        if node.group.children[index] == NOT_EXIST:
                            child = self._create_child(current, index)
                            child.flags |= (LEAF|HAS_PARTICLE)
                            child.group.data.pid = i
                            break # particle done

                        else: # internal node, travel down
                            current = node.group.children[index]

        # calculate node moments
        self._update_moments(ROOT, ROOT_SIBLING)

        if phd._in_parallel:
            # export top tree leaf moments and
            # recalculate node moments
            self._exchange_toptree_leafs()

        self.m = NULL
        for k in range(self.dim):
            self.x[k] = NULL

    cdef void _update_moments(self, int current, int sibling):
        """Recursively update moments of each local node. As a by
        product we collect the first child and sibling of each
        node, which allows for efficient tree walking.

        Parameters
        ----------
        current : int
            Gravity node index.

        sibling : int
            Sibling of parent of current.

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

            # toptree leafs may not have particles
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

    cdef void _update_toptree_moments(self, int current):
        """Recursively update toptree moments. Only toptree
        moments are calculated because bottom tree moments
        are correct because they only depend on local
        particles.

        Parameters
        ----------
        current : int
            Gravity node index.

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
                self._update_toptree_moments(ind)
                mass += child.group.data.mass
                for k in range(self.dim):
                    com[k] += child.group.data.mass*\
                            child.group.data.com[k]

                # next child
                ind = child.group.data.next_sibling

            if(mass):
                for k in range(self.dim):
                    com[k] /= mass

            node.group.data.mass = mass
            for k in range(self.dim):
                node.group.data.com[k] = com[k]

    cdef void _exchange_toptree_leafs(self):
        """Initially toptree moments are incorrect after local
        tree construction and moment calculation. To finalize
        the tree exchange toptree leaf moments and recalculate
        toptree moments.
        """
        cdef int i, j
        cdef Node *node
        cdef np.float64_t* comx[3]

        cdef LongArray proc   = self.toptree_leafs.get_carray("proc")
        cdef LongArray maps   = self.toptree_leafs.get_carray("map")
        cdef DoubleArray mass = self.toptree_leafs.get_carray("mass")

        self.toptree_leafs.pointer_groups(comx,
                self.toptree_leafs.carray_named_groups["com"])

        # collect toptree leaf moments belonging to our processor
        for i in range(self.toptree_leafs.get_carray_size()):
            if proc.data[i] == phd._rank:

                # copy our moments
                node = &self.nodes.array[maps.data[i]]
                for j in range(self.dim):
                    comx[j][i] = node.group.data.com[j]
                mass.data[i] = node.group.data.mass

        # exchange toptree leaf moments
        for field in self.toptree_leafs.carray_named_groups["moments"]:
            phd._comm.Allgatherv(MPI.IN_PLACE,
                    [self.toptree_leafs[field], self.send_cnts,
                        self.send_disp, MPI.DOUBLE])

        # copy imported toptree leaf moments to tree
        for i in range(self.toptree_leafs.get_carray_size()):
            if proc.data[i] != phd._rank:

                # update moments
                node = &self.nodes.array[maps.data[i]]
                for j in range(self.dim):
                    node.group.data.com[j] = comx[j][i]
                node.group.data.mass = mass.data[i]

        # recalculate toptree moments
        self._update_toptree_moments(ROOT)

    def walk(self, CarrayContainer particles):
        """Walk the tree calculating accerlerations.

        Parameters
        ----------
        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        self.export_interaction.initialize_particles(particles)
        if phd._in_parallel:
            self._parallel_walk(self.export_interaction, particles)
        else:
            self._serial_walk(self.export_interaction, particles)

    cdef void _serial_walk(self, Interaction interaction, CarrayContainer particles):
        """Walk the tree calculating interactions. Interactions can be any
        calculation between particles.

        Parameters
        ----------
        interaction : Interaction
            Computation for particle and node.

        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

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
        """Walk tree calculating interactions for particle that are
        imported to this process.

        Parameters
        ----------
        interaction : Interaction
            Computation for particle and node.

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
                        # skip toptree leaf that does not belong
                        # to this processor
                        index = node.group.data.next_sibling

                    else: # calculate particle particle interaction
                        interaction.interact(node)
                        index = node.group.data.next_sibling

                else: # node is not leaf
                    if(node.flags & SKIP_BRANCH):
                        # we can skip branch if node only depends
                        # on nodes from other processors
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
        """Walk tree calculating interactions for particle on this
        process. Particle are also flagged for export.

        Parameters
        ----------
        interaction : Interaction
            Computation for particle and node.

        """
        cdef Node *node
        cdef int index, i, node_pid

        cdef LongArray pid = self.toptree_leafs.get_carray("proc")

        # loop through each real praticle
        while(interaction.process_particle()):

            # start at root or next node from previous walk
            index = interaction.start_node_index()
            while(index != -1):

                node = &self.nodes.array[index]
                if(node.flags & LEAF):
                    if(node.flags & TOP_TREE_LEAF_REMOTE):
                        if(interaction.splitter.split(node)):

                            # node opend check if particle alreay flagged
                            node_pid = pid.data[self.toptree_leaf_map[index]]
                            if interaction.flag_pid[node_pid]:
                                index = node.group.data.next_sibling
                            else:

                                # particle exported to pid
                                interaction.flag_pid[node_pid] = 1

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

            # ready for next particle
            interaction.particle_finished()

    cdef void _parallel_walk(self, Interaction interaction, CarrayContainer particles):
        """Walk the tree calculating interactions. Interactions can be any
        calculation between particles.

        Parameters
        ----------
        interaction : Interaction
            Computation for particle and node.

        particles : CarrayContainer
            Class that holds all information pertaining to the particles.

        """
        cdef int i
        cdef long num_import
        cdef int local_done, global_done
        cdef np.ndarray loc_done, glb_done

        loc_done = np.zeros(1, dtype=np.int32)
        glb_done = np.zeros(1, dtype=np.int32)

        # clear out buffers
        self.buffer_size = 0

        # setup local particles for walk
        self.export_interaction.initialize_particles(particles)
        while True:

            # reset buffers
            self.buffer_size = 0
            self.indices.resize(self.buffer_size)
            self.buffer_import.resize(self.buffer_size)
            self.buffer_export.resize(self.buffer_size)

            # reset import/export counts
            for i in range(phd._size):
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
                self.buffer_export.copy(particles, self.indices,
                        particles.carray_named_groups["gravity-walk-export"])

            # send number of exports to all processors
            phd._comm.Alltoall([self.send_cnts, MPI.INT],
                    [self.recv_cnts, MPI.INT])

            # how many incoming particles
            num_import = 0
            for i in range(phd._size):
                num_import += self.recv_cnts[i]

            # create displacement arrays 
            self.send_disp[0] = self.recv_disp[0] = 0
            for i in range(1, phd._size):
                self.send_disp[i] = self.send_cnts[i-1] + self.send_disp[i-1]
                self.recv_disp[i] = self.recv_cnts[i-1] + self.recv_disp[i-1]

            # resize to fit number of particle incoming
            self.buffer_import.resize(num_import)

            # send our particles / recieve particles 
            exchange_particles(self.buffer_import, self.buffer_export,
                    self.send_cnts, self.recv_cnts,
                    0, phd._comm,
                    particles.carray_named_groups["gravity-walk-export"],
                    self.send_disp, self.recv_disp)

            # walk imported particles
            self.import_interaction.initialize_particles(self.buffer_import, False)
            self._import_walk(self.import_interaction)

            # recieve back our paritcles / send back particles
            exchange_particles(self.buffer_export, self.buffer_import,
                    self.recv_cnts, self.send_cnts,
                    0, phd._comm,
                    particles.carray_named_groups["gravity-walk-import"],
                    self.recv_disp, self.send_disp)

            # copy back our data
            particles.add(self.buffer_export, self.indices,
                    particles.carray_named_groups["gravity-walk-import"])

            # let all processors know if walk is complete 
            glb_done[0] = 0
            loc_done[0] = self.export_interaction.done_processing()
            phd._comm.Allreduce([loc_done, MPI.INT], [glb_done, MPI.INT], op=MPI.SUM)

            # if all processors tree walks are done exit
            if glb_done[0] == phd._size:
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
        cdef LongArray maps = self.toptree_leafs.get_carray('map')
        cdef LongArray proc = self.toptree_leafs.get_carray('proc')

        for i in range(self.toptree_leafs.get_carray_size()):
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
