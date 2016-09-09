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
# from Thermodynamics cimport LatentHeat
# from FluxDivergence cimport scalar_flux_divergence

import cython

cdef class ScalarDiffusion:
    def __init__(self, namelist):
        return

    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1):
        self.flux_M1 = np.zeros((M1.nv_scalars,Gr.nzg,),dtype=np.double,order='c')
        self.tendencies_M1 = np.zeros((M1.nv_scalars,Gr.nzg,),dtype=np.double,order='c')

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
            Py_ssize_t k, n, flux_shift, shift
            double [:] rho0 = Ref.rho0
            double [:] alpha0_half = Ref.alpha0_half
            double [:,:] flux = self.flux_M1
            double [:,:] M1_tendencies = M1.tendencies
            double [:,:] tendencies = self.tendencies_M1#np.zeros(shape=M1.tendencies.shape, dtype=np.double, order='c')
            double [:,:] nu = SGS.diffusivity_M1
            Py_ssize_t scalar_count = 0
            double dzi = 1/Gr.dz

            Py_ssize_t th_index = M1.name_index['th']


        # self.flux_M1 = np.zeros((M1.nv_scalars*Gr.nzg,),dtype=np.double,order='c')
        # self.tendencies_M1 = np.zeros((M1.nv_scalars*Gr.nzg,),dtype=np.double,order='c')

        # with nogil:
        if 1 == 1:
            for n in xrange(M1.nv):
                if M1.var_type[n] == 1:
                    # print('SD: M1.n', n)
                    for k in xrange(Gr.nzg):
                        flux_shift = scalar_count
                        shift = n*Gr.nzg + k
                        # print('flux_shift', flux_shift, flux.size, scalar_count, M1.nv_scalars, Gr.nzg)
                        # print('shift', shift, M1.values.size)
                        # print('k', k, rho0.size)
                        flux[flux_shift,k] = rho0[k] * 0.5*(nu[n,k]+nu[n,k+1]) * (M1.values[n,k+1] - M1.values[n,k]) * dzi
                        # a = rho0[k] * 0.5*(nu[k]+nu[k+1]) * (M1.values[shift+1] - M1.values[shift]) * dzi

                    for k in xrange(1,Gr.nzg-1):
                        flux_shift = scalar_count
                        tendencies[flux_shift,k] = - alpha0_half[k] * (flux[flux_shift,k] - flux[flux_shift,k-1]) * dzi
                        M1_tendencies[n,k] += tendencies[flux_shift,k]
                        # print('SD: M1_tendencies[', n, k']: ', M1_tendencies[n,k])

                    scalar_count += 1

        # print('SD: M1_tendencies[phi=s,k=10]: ', M1_tendencies[th_index+10], np.amax(M1_tendencies))