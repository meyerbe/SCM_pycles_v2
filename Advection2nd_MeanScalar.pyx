__author__ = 'pressel'

import numpy as np
cimport numpy as np

import cython
from cython.parallel import prange

class ScalarAdvection2nd:
    def __init__(self, grid):
        self.fluxx = None
        self.fluxy = None
        self.fluxz = None
        self.tendency = None

        return

    def initialize(self, nml, grid, scalars):
        self.fluxx = np.zeros((grid.nxl, grid.nyl, grid.nzl, scalars.ndof))
        self.fluxy = np.zeros((grid.nxl, grid.nyl, grid.nzl, scalars.ndof))
        self.fluxz = np.zeros((grid.nxl, grid.nyl, grid.nzl, scalars.ndof))
        self.tendencies = np.zeros((grid.nxl, grid.nyl, grid.nzl, scalars.ndof))

        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def update(self, grid, basicstate ,scalars, velocities):

        cdef double [:,:,:,:] phi = scalars.values[:,:,:,:]
        cdef double [:,:,:,:] scalar_tendencies = scalars.tendencies[:,:,:,:]

        cdef int updof = velocities.get_dof('u')
        cdef int vpdof = velocities.get_dof('v')
        cdef int wpdof = velocities.get_dof('w')

        cdef double [:,:,:] u = velocities.values[:,:,:,updof]
        cdef double [:,:,:] v = velocities.values[:,:,:,vpdof]
        cdef double [:,:,:] w = velocities.values[:,:,:,wpdof]

        cdef double [:,:,:,:] fluxx =  self.fluxx
        cdef double [:,:,:,:] fluxy =  self.fluxy
        cdef double [:,:,:,:] fluxz =  self.fluxz
        cdef double [:,:,:,:] tendencies = self.tendencies

        cdef double [:] alpha0 = basicstate.alpha0[:]
        cdef double [:] rho0 = 1.0/basicstate.alpha0[:]
        cdef double [:] zi = grid.zi

        cdef int nxl = grid.nxl
        cdef int nyl = grid.nyl
        cdef int nzl = grid.nzl
        cdef int ndof = scalars.ndof

        cdef double dxi = 1.0/grid.dx
        cdef double dyi = 1.0/grid.dy


        cdef int i, j, k, n
        with nogil:
            for i in prange(nxl-1,schedule='static'):
                for j in xrange(nyl-1):
                    for k in xrange(nzl-1):
                        for n in xrange(ndof):
                            fluxx[i,j,k,n] =  u[i,j,k] * (phi[i,j,k,n] + phi[i+1,j,k,n]) * 0.5
                            fluxy[i,j,k,n] =  v[i,j,k] * (phi[i,j,k,n] + phi[i,j+1,k,n]) * 0.5
                            fluxz[i,j,k,n] =  0.5 * (rho0[k] + rho0[k+1])*w[i,j,k] * (phi[i,j,k,n] + phi[i,j,k+1,n]) * 0.5

            for i in prange(1,nxl-1,schedule='static'):
                for j in xrange(1,nyl-1):
                    for k in xrange(1,nzl-1):
                        for n in xrange(ndof):
                            tendencies[i,j,k,n] = ((-(fluxx[i,j,k,n]-fluxx[i-1,j,k,n]))*dxi
                                                   -((fluxy[i,j,k,n] - fluxy[i,j-1,k,n]))*dyi
                                                   -((fluxz[i,j,k,n]-fluxz[i,j,k-1,n]))/(zi[k]-zi[k-1])*alpha0[k])
                            scalar_tendencies[i,j,k,n] = scalar_tendencies[i,j,k,n] + tendencies[i,j,k,n]



        return

    def output(self,grid,scalars,io,comm):

        return

