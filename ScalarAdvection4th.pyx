__author__ = 'pressel'

import numpy as np
cimport numpy as np

import pylab as plt
import cython
from cython.parallel import prange
from libc.math cimport fabs
import sys
import time

from SkamarockWickerInterpolation cimport fourth_order

include 'Parameters.pxi'


class ScalarAdvection4th:

    def __init__(self,grid):
        self.fluxx = None
        self.fluxy = None
        self.fluxz = None
        self.tendencies = None
        self.mp_preserving = None
        return

    def initialize(self, nml, grid, scalars):
        self.fluxx = np.zeros((grid.nxl, grid.nyl, grid.nzl, scalars.ndof),dtype=np.double,order='c')
        self.fluxy = np.zeros((grid.nxl, grid.nyl, grid.nzl, scalars.ndof),dtype=np.double,order='c')
        self.fluxz = np.zeros((grid.nxl, grid.nyl, grid.nzl, scalars.ndof),dtype=np.double,order='c')
        self.tendencies = np.zeros((grid.nxl, grid.nyl, grid.nzl, scalars.ndof),dtype=np.double,order='c')


        try:
            self.mp_preserving = nml.numerics['mp_preserving']
        except:
            self.mp_preserving = False


        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def update(self, grid, basicstate ,scalars, velocities):

        cdef bint mp_preserving = self.mp_preserving

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
        cdef double [:] zc = grid.zc

        cdef int nxl = grid.nxl
        cdef int nyl = grid.nyl
        cdef int nzl = grid.nzl
        cdef int gw = grid.gw
        cdef int ndof = scalars.ndof

        cdef double dxi = 1.0/grid.dx
        cdef double dyi = 1.0/grid.dy

        cdef int i, j, k, n, ii
        cdef double rhok = 0.0
        cdef double rhoi = 0.0


        with nogil:
            for i in prange(gw-1,nxl-gw,schedule='static'):
                for j in xrange(gw-1,nyl-gw):
                    for k in xrange(gw-1,nzl-gw):
                        rhok = rho0[k]
                        rhoi =(0.5*rho0[k]+0.5*rho0[k+1])
                        for n in xrange(ndof):


                            '''calculate the fluxes'''
                            fluxx[i,j,k,n] = u[i,j,k] * fourth_order(phi[i-1,j,k,n],phi[i,j,k,n],
                                                                    phi[i+1,j,k,n],phi[i+2,j,k,n])


                            fluxy[i,j,k,n] = v[i,j,k] * fourth_order(phi[i,j-1,k,n],phi[i,j,k,n],
                                                                    phi[i,j+1,k,n],phi[i,j+2,k,n])


                            fluxz[i,j,k,n] = w[i,j,k] * fourth_order(phi[i,j,k-1,n],phi[i,j,k,n],
                                                                    phi[i,j,k+1,n],phi[i,j,k+2,n])*rhoi

            for i in prange(gw,nxl-gw,schedule='static'):
                for j in xrange(gw,nyl-gw):
                    for k in xrange(gw,nzl-gw):
                        for n in xrange(ndof):
                            tendencies[i,j,k,n] = -(fluxx[i,j,k,n] - fluxx[i-1,j,k,n])*dxi - (fluxy[i,j,k,n] - fluxy[i,j-1,k,n])*dyi   - (fluxz[i,j,k,n] - fluxz[i,j,k-1,n])/(zi[k]-zi[k-1])/rho0[k]
                            scalar_tendencies[i,j,k,n] = scalar_tendencies[i,j,k,n] + tendencies[i,j,k,n]

        return

    def output(self,grid,scalars,io,comm):

        return