#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True


from Grid cimport Grid
from ReferenceState cimport ReferenceState
from TimeStepping cimport TimeStepping
cimport PrognosticVariables
# from PrognosticVariables cimport MeanVariables
# cimport DiagnosticVariables
cimport SGS
from NetCDFIO cimport NetCDFIO_Stats

import numpy as np
cimport numpy as np
import pylab as plt
import matplotlib
matplotlib.rc('xtick', labelsize=5)
matplotlib.rc('ytick', labelsize=8)


cdef class Diffusion:
    def __init__(self):
        return

    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1):
        self.flux_M1 = np.zeros((M1.nv,Gr.nzg,),dtype=np.double,order='c')
        self.tendencies_M1 = np.zeros((M1.nv,Gr.nzg,),dtype=np.double,order='c')

        self.grad = np.zeros((M1.nv,Gr.nzg,),dtype=np.double,order='c')
        return


    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, SGS):
        '''
        Update method for diffusion class, based on a second order finite difference scheme. The method should
        only be called following a call to update method for the SGS class.
        :param Gr: Grid class
        :param RS: ReferenceState class
        :param PV: PrognosticVariables class
        :param SGS: Subgrid Scale Diffusion
        :return:
        '''
        self.update_M1(Gr, Ref, M1, SGS)

        return


    cpdef stats_io(self):
        '''
        Statistical output for ScalarDiffusion class.
        :param Gr: Grid class
        :param RS: ReferenceState class
        :param PV: PrognosticVariables class
        :param DV: DiagnosticVariables class
        :param NS: NetCDFIO_Stats class
        :return:
        '''

        return






    cpdef update_M1(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, SGS):
        cdef:
            Py_ssize_t k, n
            Py_ssize_t scalar_count = 0
            double [:] rho0 = Ref.rho0
            double [:] alpha0 = Ref.alpha0
            double [:] rho0_half = Ref.rho0_half
            # double [:] alpha0_half = Ref.alpha0_half
            double [:,:] flux = self.flux_M1
            double [:,:] M1_tendencies = M1.tendencies
            double [:,:] tendencies = self.tendencies_M1#np.zeros(shape=M1.tendencies.shape, dtype=np.double, order='c')
            double [:,:] grad = self.grad #np.zeros((M1.nv,Gr.nzg),dtype=np.double,order='c')
            double [:,:] visc = SGS.viscosity_M1
            double [:,:] diff = SGS.diffusivity_M1
            double dzi = Gr.dzi

            Py_ssize_t w_shift = M1.name_index['w']

        # self.flux_M1 = np.zeros((M1.nv_scalars*Gr.nzg,),dtype=np.double,order='c')
        # self.tendencies_M1 = np.zeros((M1.nv_scalars*Gr.nzg,),dtype=np.double,order='c')

        with nogil:
        # if 1 == 1:
            for n in xrange(M1.nv):
                for k in xrange(1,Gr.nzg):
                    grad[n,k] = dzi*(M1.values[n,k]-M1.values[n,k-1])      # on half-grid
                    # grad[n,k] = dzi2*(M1.values[n,k+1]-M1.values[n,k-1])

                if M1.var_type[n] == 0:
                    for k in xrange(1,Gr.nzg):
                        flux[n,k] = - rho0_half[k] * visc[n,k] * grad[n,k]
                        # flux[n,k] = - rho0[k] * visc[n,k] * grad[n,k]
                elif M1.var_type[n] == 1:
                    # print('scalar count', scalar_count, diff.shape)
                    for k in xrange(1,Gr.nzg-1):
                        flux[n,k] = - rho0_half[k] * diff[scalar_count,k] * grad[n,k]
                        # flux[n,k] = - rho0[k] * diff[scalar_count,k] * grad[n,k]
                    scalar_count += 1

            for n in xrange(M1.nv):
                if n >= w_shift:
                    for k in xrange(2,Gr.nzg-2):
                        tendencies[n,k] = - alpha0[k] * (flux[n,k+1]-flux[n,k])*dzi
                        # tendencies[n,k] = - alpha0[k] * (flux[n,k+1]-flux[n,k-1])*dzi2
                else:
                    for k in xrange(2,Gr.nzg-2):
                        tendencies[n,k] = - 0.5 * alpha0[k] * (flux[n,k+1]-flux[n,k])*dzi
                        # tendencies[n,k] = - 0.5 * alpha0[k] * (flux[n,k+1]-flux[n,k-1])*dzi2


        with nogil:
            for n in xrange(M1.nv):
                for k in xrange(Gr.nzg):
                    M1_tendencies[n,k] += tendencies[n,k]


        # print(np.where(self.tendencies_M1 != M1_tendencies))
        if np.isnan(self.tendencies_M1).any():
            print('???? NAN in Diff Tendencies M1')
            if np.isnan(self.tendencies_M1[3,:]).any():
                print('???? NAN in Diff Tendencies M1, 3')
                if np.isnan(grad).any():
                    print('???? NAN in grad')       # not true in first round (t=0.0), but afterward (t>=10.0)
                if np.isnan(diff).any():
                    print('???? NAN in diff')       # not true


        return





    def plot(self, Grid Gr, TimeStepping TS, PrognosticVariables.MeanVariables M1):
        if np.isnan(self.tendencies_M1).any():
            print('!!!!! NAN in Diff Tendencies M1')
            if np.isnan(self.tendencies_M1[0,:]).any():
                print('!!!!! NAN in Diff Tendencies M1, 0')
            if np.isnan(self.tendencies_M1[1,:]).any():
                print('!!!!! NAN in Diff Tendencies M1, 1')
            if np.isnan(self.tendencies_M1[2,:]).any():
                print('!!!!! NAN in Diff Tendencies M1, 2')
            if np.isnan(self.tendencies_M1[3,:]).any():
                print('!!!!! NAN in Diff Tendencies M1, 3')
        if np.isnan(self.flux_M1).any():
            print('!!!!! NAN in Diff Fluxes M1')

        # plt.figure(1,figsize=(12,5))
        # # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        # plt.subplot(1,4,1)
        # plt.plot(self.grad[0,:], Gr.z, 'x-', label='grad u')
        # plt.plot(self.grad[1,:], Gr.z, 'x-', label='grad v')
        # plt.plot(self.grad[2,:], Gr.z, 'x-', label='grad w')
        # plt.title('grad u, v, w')
        # plt.legend(fontsize=8)
        # plt.subplot(1,4,2)
        # plt.plot(self.tendencies_M1[0,:], Gr.z, 'x-', label='tend u')
        # plt.plot(self.tendencies_M1[1,:], Gr.z, 'x-', label='tend v')
        # plt.plot(self.tendencies_M1[2,:], Gr.z, 'x-', label='tend w')
        # plt.title('M1 Tendencies')
        # plt.legend(fontsize=8)
        # plt.subplot(1,4,3)
        # plt.plot(self.grad[3,:], Gr.z, 'x-', label='tend th')
        # plt.plot(self.grad[3,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        # plt.plot(self.grad[3,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        # plt.title('grad th')
        # plt.legend(fontsize=8)
        # plt.subplot(1,4,4)
        # plt.plot(self.tendencies_M1[3,:], Gr.z, 'x-', label='tend th')
        # plt.plot(self.tendencies_M1[3,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        # plt.plot(self.tendencies_M1[3,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        # plt.title('M1 Tendencies')
        # plt.legend(fontsize=8)
        # # plt.show()
        # plt.savefig('./figs/Diffusion_' + np.str(np.int(TS.t)) + '.png')
        # plt.close()



        time = TS.dt*np.ones(shape=Gr.z.shape[0])
        plt.figure(1,figsize=(13,15))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(3,4,1)
        plt.plot(M1.values[0,:], Gr.z, 'x-', label='u')
        plt.plot(M1.values[0,0:Gr.gw], Gr.z[0:Gr.gw], 'rx', label='u')
        plt.plot(M1.values[0,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        plt.xlabel
        plt.title(' u')
        plt.subplot(3,4,5)
        plt.plot(M1.values[1,:], Gr.z, 'x-', label=' v')
        plt.plot(M1.values[1,0:Gr.gw], Gr.z[0:Gr.gw], 'rx', label='u')
        plt.plot(M1.values[1,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        plt.title(' v')
        plt.subplot(3,4,9)
        plt.plot(M1.values[2,:], Gr.z, 'x-', label=' w')
        plt.plot(M1.values[2,0:Gr.gw], Gr.z[0:Gr.gw], 'rx', label='u')
        plt.plot(M1.values[2,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        plt.title(' w')
        plt.subplot(3,4,2)
        plt.plot(self.grad[0,:], Gr.z, 'x-', label='grad u')
        plt.plot(self.grad[0,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(self.grad[0,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        plt.title('grad u')
        plt.subplot(3,4,6)
        plt.plot(self.grad[1,:], Gr.z, 'x-', label='grad v')
        plt.plot(self.grad[1,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(self.grad[1,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        plt.title('grad v')
        plt.subplot(3,4,10)
        plt.plot(self.grad[2,:], Gr.z, 'x-', label='grad w')
        plt.plot(self.grad[2,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(self.grad[2,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        plt.title('grad w')
        plt.subplot(3,4,3)
        plt.plot(self.tendencies_M1[0,:], Gr.z, 'x-', label='self.tend u')
        plt.plot(self.tendencies_M1[0,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(self.tendencies_M1[0,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        # plt.plot(M1.tendencies[0,:], Gr.z, '-', label='M1.tend')
        plt.legend(fontsize=8)
        plt.title('u Tendency')
        plt.subplot(3,4,7)
        plt.plot(self.tendencies_M1[1,:], Gr.z, 'x-', label='tend v')
        plt.plot(self.tendencies_M1[1,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(self.tendencies_M1[1,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        # plt.plot(M1.tendencies[1,:], Gr.z, '-', label='M1.tend')
        plt.title('v Tendency')
        plt.subplot(3,4,11)
        plt.plot(self.tendencies_M1[2,:], Gr.z, 'x-', label='tend w')
        plt.plot(self.tendencies_M1[2,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(self.tendencies_M1[2,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        # plt.plot(M1.tendencies[2,:], Gr.z, '-', label='M1.tend')
        plt.title('w Tendency')
        plt.subplot(3,4,4)
        plt.plot(self.tendencies_M1[0,:]*time[:], Gr.z, 'x-', label='self.tend u')
        plt.plot(self.tendencies_M1[0,0:Gr.gw]*time[0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(self.tendencies_M1[0,Gr.gw+Gr.nz:Gr.nzg]*time[Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        # plt.plot(M1.tendencies[0,:], Gr.z, '-', label='M1.tend')
        plt.legend(fontsize=8)
        plt.title('u Tendency * dt')
        plt.subplot(3,4,8)
        plt.plot(self.tendencies_M1[1,:]*time, Gr.z, 'x-', label='tend v')
        plt.plot(self.tendencies_M1[1,0:Gr.gw]*time[0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(self.tendencies_M1[1,Gr.gw+Gr.nz:Gr.nzg]*time[Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        # plt.plot(M1.tendencies[1,:], Gr.z, '-', label='M1.tend')
        plt.title('v Tendency * dt')
        plt.subplot(3,4,12)
        plt.plot(self.tendencies_M1[2,:]*time, Gr.z, 'x-', label='tend w')
        plt.plot(self.tendencies_M1[2,0:Gr.gw]*time[0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(self.tendencies_M1[2,Gr.gw+Gr.nz:Gr.nzg]*time[Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        # plt.plot(M1.tendencies[2,:], Gr.z, '-', label='M1.tend')
        plt.title('w Tendency * dt')
        plt.savefig('./figs/Diffusion_vel_' + np.str(np.int(TS.t)) + '.png')
        plt.close()

        # plt.figure(1,figsize=(12,6))
        # # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        # plt.subplot(1,4,1)
        # plt.plot(self.tendencies_M1[0,:], Gr.z, 'x-', label='tend u')
        # plt.plot(self.tendencies_M1[1,:], Gr.z, 'x-', label='tend v')
        # plt.plot(self.tendencies_M1[2,:], Gr.z, 'x-', label='tend w')
        # plt.title('M1 Tendencies')
        # plt.legend()
        # plt.subplot(1,4,2)
        # plt.plot(self.flux_M1[0,:], Gr.z, 'x-', label='flux u')
        # plt.plot(self.flux_M1[1,:], Gr.z, 'x-', label='flux v')
        # plt.plot(self.flux_M1[2,:], Gr.z, 'x-', label='flux w')
        # plt.title('M1 Fluxes')
        # plt.legend()
        # plt.subplot(1,4,3)
        # plt.plot(self.tendencies_M1[3,:], Gr.z, 'x-', label='tend th')
        # plt.title('M1 Tendencies')
        # plt.legend()
        # plt.subplot(1,4,4)
        # plt.plot(self.flux_M1[3,:], Gr.z, 'x-', label='flux th')
        # plt.title('M1 Fluxes')
        # plt.legend()
        # # plt.show()
        # plt.savefig('./figs/Diffusion_' + np.str(np.int(TS.t)) + '.png')
        # plt.close()
        return