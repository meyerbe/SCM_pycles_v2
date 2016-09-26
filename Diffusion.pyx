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

# from FluxDivergence cimport momentum_flux_divergence

cdef class Diffusion:
    def __init__(self):
        return

    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1):
        self.flux_M1 = np.zeros((M1.nv,Gr.nzg,),dtype=np.double,order='c')
        self.tendencies_M1 = np.zeros((M1.nv,Gr.nzg,),dtype=np.double,order='c')

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
            # double [:] rho0_half = Ref.rho0_half
            # double [:] alpha0_half = Ref.alpha0_half
            double [:,:] flux = self.flux_M1
            double [:,:] M1_tendencies = M1.tendencies
            double [:,:] tendencies = self.tendencies_M1#np.zeros(shape=M1.tendencies.shape, dtype=np.double, order='c')
            double [:,:] grad = np.zeros((M1.nv,Gr.nzg),dtype=np.double,order='c')
            double [:,:] visc = SGS.viscosity_M1
            double [:,:] diff = SGS.diffusivity_M1
            double dzi = 1/Gr.dz
            double dzi2 = 1/(2*Gr.dz)

            Py_ssize_t w_shift = M1.name_index['w']

        # self.flux_M1 = np.zeros((M1.nv_scalars*Gr.nzg,),dtype=np.double,order='c')
        # self.tendencies_M1 = np.zeros((M1.nv_scalars*Gr.nzg,),dtype=np.double,order='c')

        with nogil:
        # if 1 == 1:
            for n in xrange(M1.nv):
                for k in xrange(1,Gr.nzg-1):
                    grad[n,k] = dzi2*(M1.values[n,k+1]-M1.values[n,k-1])

                if M1.var_type[n] == 0:
                    for k in xrange(1,Gr.nzg-1):
                        # vgrad[ijk] = (v[ijk + sp1] - v[ijk])*dxi;
                        # strain_rate[shift + ijk] = 0.5 * (vgrad[shift_v1 + ijk] + vgrad[shift_v2 + ijk]) ;
                        # flux[ijk] = -2.0 * strain_rate[ijk] * viscosity[ijk + stencil[i1]] * rho0_half[k];
                        flux[n,k] = - rho0[k] * visc[n,k] * grad[n,k]
                elif M1.var_type[n] == 1:
                    # print('scalar count', scalar_count, diff.shape)
                    for k in xrange(1,Gr.nzg-1):
                        flux[n,k] = - rho0[k] * diff[scalar_count,k] * grad[n,k]
                    scalar_count += 1

            for n in xrange(M1.nv):
                if n >= w_shift:
                    for k in xrange(2,Gr.nzg-2):
                        tendencies[n,k] = - alpha0[n] * (flux[n,k+1]-flux[n,k-1])*dzi2
                else:
                    for k in xrange(2,Gr.nzg-2):
                        tendencies[n,k] = - 0.5 * alpha0[n] * (flux[n,k+1]-flux[n,k-1])*dzi2


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





    def plot(self, Grid Gr, TimeStepping TS):
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

        plt.figure(1,figsize=(15,7))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(self.tendencies_M1[0,:], Gr.z, 'x-', label='tend u')
        plt.plot(self.tendencies_M1[1,:], Gr.z, 'x-', label='tend v')
        plt.plot(self.tendencies_M1[2,:], Gr.z, 'x-', label='tend w')
        plt.title('M1 Tendencies')
        plt.legend()
        plt.subplot(1,4,2)
        plt.plot(self.flux_M1[0,:], Gr.z, 'x-', label='flux u')
        plt.plot(self.flux_M1[1,:], Gr.z, 'x-', label='flux v')
        plt.plot(self.flux_M1[2,:], Gr.z, 'x-', label='flux w')
        plt.title('M1 Fluxes')
        plt.legend()
        plt.subplot(1,4,3)
        plt.plot(self.tendencies_M1[3,:], Gr.z, 'x-', label='tend th')
        plt.title('M1 Fluxes')
        plt.legend()
        plt.subplot(1,4,4)
        plt.plot(self.flux_M1[3,:], Gr.z, 'x-', label='flux th')
        plt.title('M1 Fluxes')
        plt.legend()
        # plt.show()
        plt.savefig('./figs/Diffusion_' + np.str(TS.t) + '.png')
        plt.close()
        return