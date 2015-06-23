__author__ = 'pressel'

import numpy as np
cimport numpy as np

import pylab as plt
import cython
from cython.parallel import prange

class CumulantAdvection2nd:
    def __init__(self):
        self.fluxz = None
        self.tendencies = None
        self.eddy_tendencies = None

    def initialize(self, grid, velocities_mean, cumulants):
        nzl = grid.nz
        array_dims = cumulants.shape
        self.flux_z1 = np.zeros(array_dims,dtype=np.double,order='c')
        self.flux_z2 = np.zeros(array_dims,dtype=np.double,order='c')
        self.eddy_flux_z1 = np.zeros(array_dims,dtype=np.double,order='c')
        self.eddy_flux_z2 = np.zeros(array_dims,dtype=np.double,order='c')
        self.tendencies = np.zeros(array_dims, dtype=np.double,order='c')
        self.eddy_tendencies = np.zeros(array_dims, dtype=np.double,order='c')
        #print('eddy tendencies', self.eddy_tendencies.shape, np.amax(self.eddy_tendencies))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def update(self, grid, basicstate ,scalars_mean, velocities_mean, cumulants):

        cdef int udof = velocities_mean.get_dof('u')
        cdef int vdof = velocities_mean.get_dof('v')
        cdef int wdof = velocities_mean.get_dof('w')
        cdef int thdof = scalars_mean.get_dof('potential_temperature')
        # cdef int uwdof = cumulants.get_dof('uw')
        # cdef int wwdof = cumulants.get_dof('ww')
        # cdef int utdof = cumulants.get_dof('ut')
        # cdef int wtdof = cumulants.get_dof('wt')
        # cdef int ttdof = cumulants.get_dof('tt')
        # cdef int updof = cumulants.get_dof('up')
        # cdef int vpdof = cumulants.get_dof('vp')
        # cdef int wpdof = cumulants.get_dof('wp')
        # cdef int tpdof = cumulants.get_dof('tp')
        # cdef int ppdof = cumulants.get_dof('pp')

        cdef double [:] u =  velocities_mean.values[:,udof]
        cdef double [:] v =  velocities_mean.values[:,vdof]
        cdef double [:] w =  velocities_mean.values[:,wdof]
        cdef double [:] th = scalars_mean.values[:,thdof]
        cdef double [:,:,:,:] c = cumulants.values[:,:,:,:]

        cdef double [:,:,:,:] tendencies = self.tendencies[:,:,:,:]
        cdef double [:,:,:,:] cumulant_tendencies = cumulants.tendencies[:,:,:,:]

        cdef double [:,:,:,:] flux_z1 = self.flux_z1
        cdef double [:,:,:,:] flux_z2 = self.flux_z2
        cdef double [:,:,:,:] eddy_flux_z1 = self.eddy_flux_z1
        cdef double [:,:,:,:] eddy_flux_z2 = self.eddy_flux_z2
        #cdef double [:,:,:] eddy_fluxz = np.zeros((nzl, velocities_mean.ndof),dtype=np.double,order='c')

        cdef double [:] zc = grid.zc[:]
        cdef double [:] zi = grid.zi[:]
        cdef double [:] alpha0 = basicstate.alpha0[:]

        cdef double dxi = 1.0/grid.dx
        cdef double dyi = 1.0/grid.dy

        cdef int nxl = grid.nxl
        cdef int nyl = grid.nyl
        cdef int nzl = grid.nz      # !!!! nzl = nz, nxl = nx
        cdef int ndof = cumulants.ndof

        cdef int i,j,k1,k2,n
        cdef double alpha0_hi = 0.0
        cdef double alpha0_lo = 0.0

        # # (i)
        # d1 = 2
        # d2 = 2
        # for k1 in range(nzl):
        #     for k2 in range(nzl):
        #         flux_z1[k1,k2,d1,d2] = 0.5*(w[k1]+w[k1-1])*0.5*(c[k1,k2,d1,d2]+c[k1-1,k2,d1,d2])
        #         flux_z2[k1,k2,d1,d2] = 0.5*(w[k2]+w[k2-1])*0.5*(c[k1,k2,d1,d2]+c[k1,k2-1,d1,d2])
        #
        # # (ii)
        # d2 = 2
        # for d1 in range(ndof):
        #     if d1 != 2:
        #         for k1 in range(nzl):
        #             for k2 in range(nzl):
        #                 flux_z1[k1,k2,d1,d2] = 0.5*(w[k1]+w[k1-1])*0.5*(c[k1,k2,d1,d2]+c[k1-1,k2,d1,d2])
        #                 flux_z2[k1,k2,d1,d2] = w[k2-1]*0.5*(c[k1,k2,d1,d2]+c[k1,k2,d1,d2])
        #
        # # (iii)
        # d1 = 2
        # for d2 in range(ndof):
        #     if d2 != 2:
        #         for k1 in range(nzl):
        #             for k2 in range(nzl):
        #                 flux_z1[k1,k2,d1,d2] = w[k1-1]*0.5*(c[k1,k2,d1,d2]+c[k1-1,k2,d1,d2])
        #                 flux_z2[k1,k2,d1,d2] = 0.5*(w[k2]+w[k2-1])*0.5*(c[k1,k2,d1,d2]+c[k1,k2-1,d1,d2])
        #
        # # (iv)
        # for d1 in range(ndof):
        #     for d2 in range(ndof):
        #         if d1 != 2:
        #             if d2 != 2:
        #                 flux_z1[k1,k2,d1,d2] = w[k1-1]*0.5*(c[k1,k2,d1,d2]+c[k1-1,k2,d1,d2])
        #                 flux_z2[k1,k2,d1,d2] = w[k2-1]*0.5*(c[k1,k2,d1,d2]+c[k1,k2-1,d1,d2])

        a1 = 1
        a2 = 1
        # flux due to vertical advection by mean field
        for d1 in range(ndof):
            for d2 in range(ndof):
                if d1 == 2:
                    flux_z1[k1,k2,d1,d2] = 0.5*(w[k1]+w[k1-1])*0.5*(c[k1,k2,d1,d2]+c[k1-1,k2,d1,d2])
                else:
                    flux_z1[k1,k2,d1,d2] = w[k1-1]*0.5*(c[k1,k2,d1,d2]+c[k1-1,k2,d1,d2])
                if d2 == 2:
                    flux_z2[k1,k2,d1,d2] = 0.5*(w[k2]+w[k2-1])*0.5*(c[k1,k2,d1,d2]+c[k1,k2-1,d1,d2])
                else:
                    flux_z2[k1,k2,d1,d2] = w[k2-1]*0.5*(c[k1,k2,d1,d2]+c[k1,k2,d1,d2])



                if d1 == 2:
                    a1 = 2
                if d2 == 2:
                    a2 = 2

                for k1 in xrange(2,nzl-2):
                    for k2 in range(2,nzl-2):
                        alpha0_hi = 0.5 * (alpha0[k1] + alpha0[k1+1])
                        alpha0_lo = 0.5 * (alpha0[k1] + alpha0[k1-1])

                        tendencies[k1,k2,d1,d2] =  - a1*((flux_z1[k1+1,k2,d1,d2] - flux_z1[k1,k2,d1,d2])*alpha0[k1])/(zi[k1+1] - zi[k1])\
                                                           - a2*((flux_z2[k1,k2+1,d1,d2] - flux_z2[k1,k2,d1,d2])*alpha0[k2])/(zi[k2+1] - zi[k2])
        np = cumulants.get_dof('u p')[1]
        nt = cumulants.get_dof('u potential_temperature')[1]
        for k1 in range(nzl):
            for k2 in range(nzl):
                eddy_flux_z1[k1,k2,0,0] = c[k1-1,k2,2,0]*0.5*(u[k1]+u[k1-1])
                eddy_flux_z2[k1,k2,0,0] = c[k1,k2-1,0,2]*0.5*(u[k2]+u[k2-1])

                eddy_flux_z1[k1,k2,0,2] = c[k1-1,k2,2,2]*0.5*(u[k1]+u[k1-1])
                eddy_flux_z2[k1,k2,0,2] = c[k1,k2,0,np]

                eddy_flux_z2[k1,k2,2,0] = c[k1,k2-1,2,2]*0.5*(u[k2]+u[k2-1])
                eddy_flux_z1[k1,k2,2,0] = c[k1,k2,np,0]

                eddy_flux_z1[k1,k2,2,2] = c[k1,k2,np,2]
                eddy_flux_z2[k1,k2,2,2] = c[k1,k2,2,np]

                eddy_flux_z1[k1,k2,0,nt] = c[k1,k2,2,nt]*0.5*(u[k1]+u[k1-1])
                eddy_flux_z2[k1,k2,0,nt] = 0.5*(th[k2]+th[k2-1])*c[k1,k2,0,2]
                #eddy_flux_z1[k1,k2,nt,0] = c[k1,k2,2,nt]*0.5*(u[k1]+u[k1-1])
                #eddy_flux_z2[k1,k2,nt,0] = 0.5*(th[k2]+th[k2-1])*c[k1,k2,0,2]
                #...

                eddy_flux_z2[k1,k2,2,nt] = c[k1,k2-1,2,2]*0.5*(th[k1]+th[k1-1])
                eddy_flux_z1[k1,k2,2,nt] = c[k1,k2,np,nt]
                #eddy_flux_z2[k1,k2,nt,2] = c[k1,k2-1,2,2]*0.5*(th[k1]+th[k1-1])
                #eddy_flux_z1[k1,k2,nt,2] = c[k1,k2,np,nt]

                eddy_flux_z1[k1,k2,nt,nt] = 0.5*(th[k1]+th[k1-1])*c[k1,k2,2,nt]
                eddy_flux_z2[k1,k2,nt,nt] = 0.5*(th[k2]+th[k2-1])*c[k1,k2,nt,2]


                #eddy_flux


                    # tendencies[i,j,k,udof] =  ((-(fluxx[i+1,j,k,udof] - fluxx[i,j,k,udof]))*dxi - \
                    #                            ((fluxy[i,j+1,k,udof] - fluxy[i,j,k,udof]))*dyi \
                    #                            - ((fluxz[i,j,k+1,udof] - fluxz[i,j,k,udof])*alpha0[k])/(zi[k+1] - zi[k]))
                    #
                    # tendencies[i,j,k,vdof] =  ((-(fluxx[i+1,j,k,vdof] - fluxx[i,j,k,vdof]))*dxi - \
                    #                            ((fluxy[i,j+1,k,vdof] - fluxy[i,j,k,vdof]))*dyi \
                    #                            - ((fluxz[i,j,k+1,vdof]- fluxz[i,j,k,vdof])*alpha0[k])/(zi[k+1] - zi[k]))
                    #
                    # tendencies[i,j,k,wdof] =  ((-(fluxx[i+1,j,k,wdof] - fluxx[i,j,k,wdof]))*dxi - \
                    #                            ((fluxy[i,j+1,k,wdof] - fluxy[i,j,k,wdof]))*dyi \
                    #                            - ((fluxz[i,j,k+1,wdof]  - fluxz[i,j,k,wdof])*(alpha0_hi))/(zc[k+1] - zc[k]))



                cumulant_tendencies[k1,k2,d1,d2] = cumulant_tendencies[k1,k2,d1,d2] + tendencies[k1,k2,d1,d2]



        # with nogil:
        #     for k1 in xrange(1,nzl-1):
        #         for k2 in xrange(1,nzl-1):
        #             alpha0_hi = 0.5 * (alpha0[k1] + alpha0[k1+1])
        #             alpha0_lo = 0.5 * (alpha0[k1] + alpha0[k1-1])
        #
        #             #U Component of Momentum Fluxes
        #             fluxz[k1,udof] = w[k1-1]/alpha0_lo*(u[k1-1]+u[k1])*0.5
        #             #V Component of Momentum Fluxes
        #             #fluxz[i,j,k,vdof] = (w[i,j,k-1]/alpha0_lo+w[i,j+1,k-1]/alpha0_lo)*(v[i,j,k-1]+v[i,j,k])*0.25
        #             #W Component of Momentum Fluxes
        #             fluxz[k1,wdof] = (w[k1-1]/alpha0_lo + w[k1]/alpha0_hi)*(w[k1-1]+w[k1])*0.25
        #
        #            # eddy_fluxz[k1,udof] = (c[k1+1,k1,uwdof]+c[k1,k1,uwdof])*0.5     # u'w' >> bring u comp. on k-face
        #            # eddy_fluxz[k1,wdof] = (c[k1,k1,wwdof]+c[k1-1,k1-1,wwdof])*0.5    # w'w' >> bring on centered coord.
        #
        #     for k in xrange(2,nzl-2):
        #         alpha0_hi = 0.5 * (alpha0[k] + alpha0[k+1])
        #         alpha0_lo = 0.5 * (alpha0[k] + alpha0[k-1])
        #
        #        # eddy_tendencies[k,udof] = - ((eddy_fluxz[k,udof] - eddy_fluxz[k-1,udof])*alpha0[k])/(zi[k] - zi[k-1])
        #        # eddy_tendencies[k,wdof] = - ((eddy_fluxz[k+1,wdof] - eddy_fluxz[k,wdof])*alpha0_hi)/(zc[k+1] - zc[k])
        #
        #         tendencies[k,udof] =  - ((fluxz[k+1,udof] - fluxz[k,udof])*alpha0[k])/(zi[k+1] - zi[k])
        #         #tendencies[i,j,k,vdof] =  - ((fluxz[i,j,k+1,vdof]- fluxz[i,j,k,vdof])*alpha0[k])/(zi[k+1] - zi[k]))
        #         tendencies[k,wdof] =  - ((fluxz[k+1,wdof]  - fluxz[k,wdof])*(alpha0_hi))/(zc[k+1] - zc[k])
        #
        #
        #         for n in xrange(ndof):
        #             velocity_tendencies[k,n] = velocity_tendencies[k,n] + tendencies[k,n] + eddy_tendencies[k,n]

        return
