__author__ = 'pressel'

import numpy as np
cimport numpy as np

import pylab as plt
import cython
from cython.parallel import prange

class MomentumAdvection2nd:
    def __init__(self, grid):
        self.fluxz = None
        self.tendencies = None
        self.eddy_tendencies = None

    def initialize(self, grid, velocities_mean):
        nzl = grid.nz

        self.fluxz = np.zeros((nzl, velocities_mean.ndof),dtype=np.double,order='c')
        self.tendencies = np.zeros((nzl, velocities_mean.ndof),dtype=np.double,order='c')
        self.eddy_tendencies = np.zeros((nzl, velocities_mean.ndof),dtype=np.double,order='c')
        #print('eddy tendencies', self.eddy_tendencies.shape, np.amax(self.eddy_tendencies))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def update(self, grid, basicstate ,scalars, velocities_mean, cumulants):

        cdef int udof = velocities_mean.get_dof('u')
        cdef int vdof = velocities_mean.get_dof('v')
        cdef int wdof = velocities_mean.get_dof('w')
        cdef int uwdof = cumulants.get_dof('uw')
        cdef int wwdof = cumulants.get_dof('ww')

        cdef double [:] u =  velocities_mean.values[:,udof]
        cdef double [:] v =  velocities_mean.values[:,vdof]
        cdef double [:] w =  velocities_mean.values[:,wdof]

        cdef double [:,:,:] c = cumulants.values[:,:,:]

        cdef double [:,:] tendencies = self.tendencies[:,:]
        cdef double [:,:] velocity_tendencies = velocities_mean.tendencies[:,:]
        cdef double [:,:] eddy_tendencies = self.eddy_tendencies[:,:]

        cdef double [:,:] fluxz = self.fluxz[:,:]
        cdef double [:,:] eddy_fluxz = np.zeros(shape=fluxz.shape)

        cdef double [:] zc = grid.zc[:]
        cdef double [:] zi = grid.zi[:]
        cdef double [:] alpha0 = basicstate.alpha0[:]

        cdef double dxi = 1.0/grid.dx
        cdef double dyi = 1.0/grid.dy

        cdef int nxl = grid.nxl
        cdef int nyl = grid.nyl
        cdef int nzl = grid.nz      # !!!! nzl = nz, nxl = nx
        cdef int ndof = velocities_mean.ndof

        cdef int i,j,k,n
        cdef double alpha0_hi = 0.0
        cdef double alpha0_lo = 0.0

        with nogil:
            for k in xrange(1,nzl-1):
                alpha0_hi = 0.5 * (alpha0[k] + alpha0[k+1])
                alpha0_lo = 0.5 * (alpha0[k] + alpha0[k-1])

                #U Component of Momentum Fluxes
                fluxz[k,udof] = w[k-1]/alpha0_lo*(u[k-1]+u[k])*0.5
                #V Component of Momentum Fluxes
                #fluxz[i,j,k,vdof] = (w[i,j,k-1]/alpha0_lo+w[i,j+1,k-1]/alpha0_lo)*(v[i,j,k-1]+v[i,j,k])*0.25
                #W Component of Momentum Fluxes
                fluxz[k,wdof] = (w[k-1]/alpha0_lo + w[k]/alpha0_hi)*(w[k-1]+w[k])*0.25

                eddy_fluxz[k,udof] = (c[k+1,k,uwdof]+c[k,k,uwdof])*0.5     # u'w' >> bring u comp. on k-face
                eddy_fluxz[k,wdof] = (c[k,k,wwdof]+c[k-1,k-1,wwdof])*0.5    # w'w' >> bring on centered coord.



            for k in xrange(2,nzl-2):
                alpha0_hi = 0.5 * (alpha0[k] + alpha0[k+1])
                alpha0_lo = 0.5 * (alpha0[k] + alpha0[k-1])

                eddy_tendencies[k,udof] = - ((eddy_fluxz[k,udof] - eddy_fluxz[k-1,udof])*alpha0[k])/(zi[k] - zi[k-1])
                eddy_tendencies[k,wdof] = - ((eddy_fluxz[k+1,wdof] - eddy_fluxz[k,wdof])*alpha0_hi)/(zc[k+1] - zc[k])

                tendencies[k,udof] =  - ((fluxz[k+1,udof] - fluxz[k,udof])*alpha0[k])/(zi[k+1] - zi[k])
                #tendencies[i,j,k,vdof] =  - ((fluxz[i,j,k+1,vdof]- fluxz[i,j,k,vdof])*alpha0[k])/(zi[k+1] - zi[k]))
                tendencies[k,wdof] =  - ((fluxz[k+1,wdof]  - fluxz[k,wdof])*(alpha0_hi))/(zc[k+1] - zc[k])


                for n in xrange(ndof):
                    velocity_tendencies[k,n] = velocity_tendencies[k,n] + tendencies[k,n] + eddy_tendencies[k,n]

        return
