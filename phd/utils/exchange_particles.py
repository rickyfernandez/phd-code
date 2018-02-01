import phd
import numpy as np

def exchange_particles(particles, send_data, send_particles, recv_particles, disp, comm, fields=None,
        offset_se=None, offset_re=None):
    """Exchange particles between processes. Note you have allocate the appropriate space for
    incoming particles in the particle container.

    parameters
    ----------
    particles : object
        particle container
    send_data : ParticleArray
        particle carrays to send
    send_particles : ndarray
        number particles to send to each process
    recv_particles : ndarray
        number particles to receive from each process
    disp : int
        starting index position for incoming particles
    comm : object
        mpi controller
    fields : list
        List of fields to export
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if fields != None:
        export_fields = fields
    else:
        export_fields = particles.carrays.keys()

    if offset_se is None or offset_re is None:
        # displacements for the send and reveive buffers
        offset_se = np.zeros(size, dtype=np.int32)
        offset_re = np.zeros(size, dtype=np.int32)
        for i in xrange(1,size):
            offset_se[i] = send_particles[i-1] + offset_se[i-1]
            offset_re[i] = recv_particles[i-1] + offset_re[i-1]

    ptask = 0
    while size > (1<<ptask):
        ptask += 1

    for ngrp in xrange(1,1 << ptask):
        sendTask = rank
        recvTask = rank ^ ngrp
        if recvTask < size:
            if send_particles[recvTask] > 0 or recv_particles[recvTask] > 0:
                for prop in export_fields:

                    sendbuf=[send_data[prop],   (send_particles[recvTask], offset_se[recvTask])]
                    recvbuf=[particles[prop][disp:], (recv_particles[recvTask],
                        offset_re[recvTask])]

                    comm.Sendrecv(sendbuf=sendbuf, dest=recvTask, recvbuf=recvbuf, source=recvTask)


    # do we have particles to send to our own processor
    if send_particles[phd._rank] > 0:

        first1 = offset_se[phd._rank]
        last1 = first1 + send_particles[phd._rank]

        first2 = disp + offset_re[phd._rank]
        last2 = first2 + recv_particles[phd._rank]

        for prop in export_fields:

            particles[prop][first2:last2] = send_data[prop][first1:last1]

