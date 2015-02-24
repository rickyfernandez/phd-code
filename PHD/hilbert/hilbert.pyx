import numpy as np

cimport numpy as np
cimport cython

# turn this to a c array
cdef int *key_index_2d = [0, 1, 3, 2]
#key_index_2d[:] = [0, 1, 3, 2]

cdef int *key_index_3d = [0, 1, 7, 6, 3, 2, 4, 5]

cpdef np.uint64_t hilbert_key_2d(np.uint32_t x, np.uint32_t y, int order):

    cdef np.uint64_t key = 0
    cdef int i, xbit, ybit
    for i in xrange(order-1, -1, -1):

        xbit = 1 if x & (1 << i) else 0
        ybit = 1 if y & (1 << i) else 0

        if   xbit == 0 and ybit == 0: x, y = y, x
        elif xbit == 1 and ybit == 0: x, y = ~y, ~x

        key = (key << 2) + key_index_2d[(xbit << 1) + ybit]

    return key


cpdef np.uint64_t hilbert_key_3d(np.uint32_t x, np.uint32_t y, np.uint32_t z, int order):

    cdef np.uint64_t key = 0
    cdef int i, xbit, ybit
    for i in xrange(order-1, -1, -1):

        xbit = 1 if x & (1 << i) else 0
        ybit = 1 if y & (1 << i) else 0
        zbit = 1 if z & (1 << i) else 0

        if   xbit == 0 and ybit == 0 and zbit == 0: y, z = z, y
        elif xbit == 0 and ybit == 0 and zbit == 1: x, y = y, x
        elif xbit == 1 and ybit == 0 and zbit == 1: x, y = y, x
        elif xbit == 1 and ybit == 0 and zbit == 0: x, z = ~x, ~z
        elif xbit == 1 and ybit == 1 and zbit == 0: x, z = ~x, ~z
        elif xbit == 1 and ybit == 1 and zbit == 1: x, y = ~y, ~x
        elif xbit == 0 and ybit == 1 and zbit == 1: x, y = ~y, ~x
        elif xbit == 0 and ybit == 1 and zbit == 0: y, z = ~z, ~y

        key = (key << 3) + key_index_3d[(xbit << 2) + (ybit << 1) + zbit]

    return key
