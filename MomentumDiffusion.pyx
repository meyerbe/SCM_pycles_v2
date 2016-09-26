#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True


from Grid cimport Grid
from ReferenceState cimport ReferenceState
cimport PrognosticVariables
# from PrognosticVariables cimport MeanVariables
# cimport DiagnosticVariables
cimport SGS
from NetCDFIO cimport NetCDFIO_Stats

import numpy as np
cimport numpy as np

# from FluxDivergence cimport momentum_flux_divergence

cdef class MomentumDiffusion:
    def __init__(self):
        return

    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1):
        self.flux_M1 = np.zeros((M1.nv_velocities,Gr.nzg,),dtype=np.double,order='c')
        self.tendencies_M1 = np.zeros((M1.nv_velocities,Gr.nzg,),dtype=np.double,order='c')

        return


    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, SGS):
        '''
        Update method for scalar diffusion class, based on a second order finite difference scheme. The method should
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

        return






    cpdef update_M1(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, SGS):
        cdef:
            Py_ssize_t k, n, flux_shift, shift
            double [:] rho0 = Ref.rho0
            double [:] alpha0 = Ref.alpha0
            double [:] rho0_half = Ref.rho0_half
            double [:] alpha0_half = Ref.alpha0_half
            double [:,:] flux = self.flux_M1
            double [:,:] M1_tendencies = M1.tendencies
            double [:,:] tendencies = self.tendencies_M1#np.zeros(shape=M1.tendencies.shape, dtype=np.double, order='c')
            double [:,:] grad = np.zeros((M1.nv,Gr.nzg),dtype=np.double,order='c')
            # double [:,:] strain_rate = np.zeros((M1.nv_velocities,Gr.nzg),dtype=np.double,order='c')
            # double [:] szx = np.zeros((Gr.nzg),dtype=np.double,order='c')
            # double [:] szy = np.zeros((Gr.nzg),dtype=np.double,order='c')
            # double [:] szz = np.zeros((Gr.nzg),dtype=np.double,order='c')
            double [:,:] nu = SGS.viscosity_M1
            Py_ssize_t vel_count = 0
            double dzi = 1/Gr.dz
            double dzi2 = 1/(2*Gr.dz)


        # self.flux_M1 = np.zeros((M1.nv_scalars*Gr.nzg,),dtype=np.double,order='c')
        # self.tendencies_M1 = np.zeros((M1.nv_scalars*Gr.nzg,),dtype=np.double,order='c')

        # with nogil:
        if 1 == 1:
            for n in xrange(M1.nv):
                for k in xrange(1,Gr.nzg-1):
                    grad[n,k] = dzi2*(M1.values[n,k+1]-M1.values[n,k-1])
                # for k in xrange(1,Gr.nzg-1):
                #     strain_rate[n,k] = dzi2*(grad[k+1]-grad[k-1])

                if M1.var_type[n] == 0:
                    for k in xrange(1,Gr.nzg-1):
                        # vgrad[ijk] = (v[ijk + sp1] - v[ijk])*dxi;
                        # strain_rate[shift + ijk] = 0.5 * (vgrad[shift_v1 + ijk] + vgrad[shift_v2 + ijk]) ;
                        # flux[ijk] = -2.0 * strain_rate[ijk] * viscosity[ijk + stencil[i1]] * rho0_half[k];
                        flux_shift = n
                        flux[flux_shift,k] = - rho0[k] * nu[n,k] * grad[n,k]
            for k in xrange(2,Gr.nzg-2):
                tendencies[0,k] = - 0.5 * alpha0[k] * (flux[0,k+1]-flux[0,k-1])*dzi2
                tendencies[1,k] = - 0.5 * alpha0[k] * (flux[1,k+1]-flux[1,k-1])*dzi2
                tendencies[2,k] = - alpha0[k] * (flux[2,k+1]-flux[2,k-1])*dzi2

                # elif M1.var_type[n] == 1:
                #     for k in xrange(1,Gr.nzg-1):
                #         flux[n,k] =


        # with nogil:
        if 1 == 1:
            for n in xrange(M1.nv):
                if M1.var_type[n] == 0:
                    # print('MD: M1.n', n)
                    for k in xrange(1,Gr.nzg-1):
                        flux_shift = vel_count
                        # if flux_shift != shift:
                        #     print('shift', flux_shift, shift)

                        # vgrad[ijk] = (v[ijk + sp1] - v[ijk])*dxi;
                        # strain_rate[shift + ijk] = 0.5 * (vgrad[shift_v1 + ijk] + vgrad[shift_v2 + ijk]) ;
                        # flux[ijk] = -2.0 * strain_rate[ijk] * viscosity[ijk + stencil[i1]] * rho0_half[k];

                        # flux[flux_shift,k] = rho0[k] * 0.5*(nu[n,k]+nu[n,k+1]) * (M1.values[n,k+1] - M1.values[n,k]) * dzi
                        flux[flux_shift,k] = rho0[k] * 0.5*(nu[n,k]+nu[n,k+1]) * (M1.values[n,k+1] - M1.values[n,k]) * dzi

                    for k in xrange(1,Gr.nzg-1):
                        flux_shift = vel_count
                        tendencies[flux_shift,k] = - alpha0_half[k] * (flux[flux_shift,k] - flux[flux_shift,k-1]) * dzi
                        M1_tendencies[n,k] += tendencies[flux_shift,k]
                        # print('SD: M1_tendencies[', shift, ']: ', M1_tendencies[shift])

                    vel_count += 1

        # print('MA: M1_tendencies[u,k=10]: ', M1_tendencies[10], np.amax(M1_tendencies))