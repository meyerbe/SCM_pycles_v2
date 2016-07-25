__author__ = 'pressel'
import numpy as np
cimport numpy as np

import cython
from cython.parallel import prange

from SkamarockWickerInterpolation cimport fourth_order
from WENOInterpolation cimport vel_interp_4
class MomentumAdvection4th:
    def __init__(self,grid):
        self.fluxx = None
        self.fluxy = None
        self.fluxz = None
        self.tendencies = None
        return

    def initialize(self, grid, velocities):
        self.fluxx = np.zeros((grid.nxl, grid.nyl, grid.nzl, velocities.ndof),dtype=np.double,order='c')
        self.fluxy = np.zeros((grid.nxl, grid.nyl, grid.nzl, velocities.ndof),dtype=np.double,order='c')
        self.fluxz = np.zeros((grid.nxl, grid.nyl, grid.nzl, velocities.ndof),dtype=np.double,order='c')
        self.tendencies = np.zeros((grid.nxl, grid.nyl, grid.nzl, velocities.ndof),dtype=np.double,order='c')
        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def update(self, grid, basicstate ,scalars, velocities):
        cdef int udof = velocities.get_dof('u')
        cdef int vdof = velocities.get_dof('v')
        cdef int wdof = velocities.get_dof('w')

        cdef double [:,:,:] u =  velocities.values[:,:,:,udof]
        cdef double [:,:,:] v =  velocities.values[:,:,:,vdof]
        cdef double [:,:,:] w =  velocities.values[:,:,:,wdof]


        cdef double [:,:,:,:] tendencies = self.tendencies[:,:,:,:]
        cdef double [:,:,:,:] velocity_tendencies = velocities.tendencies[:,:,:,:]

        cdef double [:,:,:,:] fluxx = self.fluxx[:,:,:,:]
        cdef double [:,:,:,:] fluxy = self.fluxy[:,:,:,:]
        cdef double [:,:,:,:] fluxz = self.fluxz[:,:,:,:]

        cdef double [:] zc = grid.zc[:]
        cdef double [:] zi = grid.zi[:]
        cdef double [:] rho0 = 1.0/basicstate.alpha0[:]
        cdef double [:] alpha0 = basicstate.alpha0[:]

        cdef double dxi = 1.0/grid.dx
        cdef double dyi = 1.0/grid.dy

        cdef int nxl = grid.nxl
        cdef int nyl = grid.nyl
        cdef int nzl = grid.nzl
        cdef int gw = grid.gw
        cdef int ndof = velocities.ndof

        cdef double phim    #Upwind interpolated velocity when when advel < 0.0
        cdef double phip    #Upwind interpolated velocity when advelm > 0.0
        cdef double advel  #Advective velocity
        cdef double rhok   #Density at level k
        cdef int i,j,k,n
        cdef double rhoi = 0.0
        cdef double alpha_k
        with nogil:
            for i in prange(gw-1,nxl-gw,schedule='static'):
                for j in xrange(gw-1,nyl-gw):
                    for k in xrange(gw-1,nzl-gw):
                        rhok = rho0[k]
                        rhoi = (rho0[k] + rho0[k+1])*0.5

                        ''' Compute flux of u b u'''
                        advel = vel_interp_4(u[i-1,j,k],u[i,j,k],u[i+1,j,k],u[i+2,j,k]) #(u[i+1,j,k] + u[i,j,k]) * 0.5
                        fluxx[i,j,k,udof] =  rhok * vel_interp_4(u[i-1,j,k],u[i,j,k],u[i+1,j,k],u[i+2,j,k]) * advel

                        ''' Compute flux of u by v '''
                        advel = vel_interp_4(v[i-1,j,k],v[i,j,k],v[i+1,j,k],v[i+2,j,k])
                        fluxy[i,j,k,udof] =  rhok * vel_interp_4(u[i,j-1,k],u[i,j,k],u[i,j+1,k],u[i,j+2,k]) * advel

                        ''' Compute flux of u by w '''
                        advel = vel_interp_4(w[i-1,j,k],w[i,j,k],w[i+1,j,k],w[i+2,j,k])#(w[i+1,j,k] + w[i,j,k])*0.5
                        fluxz[i,j,k,udof] =  rhoi * vel_interp_4(u[i,j,k-1],u[i,j,k],u[i,j,k+1],u[i,j,k+2]) * advel

                        ''' Compute flux of v by u'''
                        #advel = (u[i,j,k] + u[i,j+1,k]) * 0.5
                        advel = vel_interp_4(u[i,j-1,k],u[i,j,k],u[i,j+1,k],u[i,j+2,k])
                        fluxx[i,j,k,vdof] =  rhok * vel_interp_4(v[i-1,j,k],v[i,j,k],v[i+1,j,k],v[i+2,j,k]) * advel

                        ''' Compute flux of v by v'''
                        #advel = (v[i,j+1,k] + v[i,j,k])*0.5
                        advel = vel_interp_4(v[i,j-1,k],v[i,j,k],v[i,j+1,k],v[i,j+2,k])
                        fluxy[i,j,k,vdof] =  rhok * vel_interp_4(v[i,j-1,k],v[i,j,k],v[i,j+1,k],v[i,j+2,k]) * advel

                        '''Compute flux of v by w'''
                        #advel = (w[i,j,k] + w[i,j+1,k])*0.5
                        advel = vel_interp_4(w[i,j-1,k],w[i,j,k],w[i,j+1,k],w[i,j+2,k])
                        fluxz[i,j,k,vdof] =  rhoi * vel_interp_4(v[i,j,k-1],v[i,j,k],v[i,j,k+1],v[i,j,k+2]) * advel

                        '''Compute flux of w by u'''
                        #advel = (u[i,j,k] + u[i,j,k+1]) * 0.5
                        advel = vel_interp_4(u[i,j,k-1],u[i,j,k],u[i,j,k+1],u[i,j,k+2])
                        fluxx[i,j,k,wdof] = rhoi * vel_interp_4(w[i-1,j,k],w[i,j,k],w[i+1,j,k],w[i+2,j,k]) * advel

                        '''Compute flux of w by v'''
                        #advel = (v[i,j,k] + v[i,j,k+1])*0.5
                        advel = vel_interp_4(v[i,j,k-1],v[i,j,k],v[i,j,k+1],v[i,j,k+2])
                        fluxy[i,j,k,wdof] =  rhoi * vel_interp_4(w[i,j-1,k],w[i,j,k],w[i,j+1,k],w[i,j+2,k]) * advel

                        '''Compute flux of w by w'''
                        #advel = (w[i,j,k+1] + w[i,j,k])*0.5
                        advel = vel_interp_4(w[i,j,k-1],w[i,j,k],w[i,j,k+1],w[i,j,k+2])
                        fluxz[i,j,k,wdof] =  rho0[k+1]* vel_interp_4(w[i,j,k-1],w[i,j,k],w[i,j,k+1],w[i,j,k+2]) * advel


            for i in prange(gw,nxl-gw,schedule='static'):
                for j in xrange(gw,nyl-gw):
                    for k in xrange(gw,nzl-gw):
                        alpha_k = 0.5 * (alpha0[k] + alpha0[k+1])

                        tendencies[i,j,k,udof] =  -(fluxx[i,j,k,udof]-fluxx[i-1,j,k,udof]) * dxi * alpha0[k] \
                                                  - (fluxy[i,j,k,udof]-fluxy[i,j-1,k,udof]) * dyi * alpha0[k] \
                                                  - (fluxz[i,j,k,udof]-fluxz[i,j,k-1,udof])/ (zi[k]-zi[k-1]) * alpha0[k]



                        tendencies[i,j,k,vdof] = -(fluxx[i,j,k,vdof]-fluxx[i-1,j,k,vdof]) * dxi * alpha0[k] \
                                                 - (fluxy[i,j,k,vdof]-fluxy[i,j-1,k,vdof]) * dyi * alpha0[k] \
                                                 - (fluxz[i,j,k,vdof]-fluxz[i,j,k-1,vdof])/ (zi[k]-zi[k-1]) * alpha0[k]



                        tendencies[i,j,k,wdof] = -(fluxx[i,j,k,wdof]-fluxx[i-1,j,k,wdof]) * dxi * alpha_k \
                                                 - (fluxy[i,j,k,wdof]-fluxy[i,j-1,k,wdof]) * dyi * alpha_k \
                                                 - (fluxz[i,j,k,wdof]-fluxz[i,j,k-1,wdof])/(zi[k]-zi[k-1])*alpha_k


                        for n in xrange(ndof):
                            velocity_tendencies[i,j,k,n] = velocity_tendencies[i,j,k,n] + tendencies[i,j,k,n]



    def output(self,grid,scalars,io,comm):

        return