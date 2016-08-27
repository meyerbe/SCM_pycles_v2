#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True


cdef class Grid:
    cdef:
        Py_ssize_t dims
        double dz
        double dzi
        double lz
        Py_ssize_t gw
        Py_ssize_t nz
        Py_ssize_t nzg
        double [:] z
        double [:] z_half
