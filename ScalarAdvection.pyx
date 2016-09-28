#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from Grid cimport Grid
from PrognosticVariables cimport MeanVariables
from PrognosticVariables cimport SecondOrderMomenta
from ReferenceState cimport ReferenceState
# # cimport DiagnosticVariables
# cimport TimeStepping
# from NetCDFIO cimport NetCDFIO_Stats

# from FluxDivergence cimport scalar_flux_divergence
# from Thermodynamics cimport LatentHeat

import numpy as np
cimport numpy as np
import sys
import pylab as plt


import cython


cdef class ScalarAdvection:
    def __init__(self, namelist):
        try:
            self.order = namelist['scalar_transport']['order']
        except:
            print('scalar_transport order not given in namelist')
            print('Killing simulation now!')
            sys.exit()
        # try:
        #     self.order_sedimentation = namelist['scalar_transport']['order_sedimentation']
        # except:
        #     self.order_sedimentation = self.order
        return


    cpdef initialize(self, Grid Gr, MeanVariables M1):
        # self.flux = np.zeros((PV.nv_scalars*Gr.dims.npg*Gr.dims,),dtype=np.double,order='c')
        self.flux = np.zeros((M1.nv_scalars,Gr.nzg),dtype=np.double,order='c')
        self.tendencies = np.zeros((M1.nv_scalars,Gr.nzg),dtype=np.double,order='c')
        return


    cpdef update(self, Grid Gr, ReferenceState Ref, MeanVariables M1):
        # (1) update tendencies for Mean Variables
        #       - only vertical advection
        if self.order == 2:
            self.update_M1_2nd(Gr, Ref, M1)
        else:
            print('scalar advection scheme not implemented')
            sys.exit()

        # (2) update tendencies for 2nd Order Momenta
        return



    cpdef update_M1_2nd(self, Grid Gr, ReferenceState Ref, MeanVariables M1):
        print('Scalar Advection M1: update 2nd')
        # (1) update tendencies for Mean Variables
        #       - only vertical advection
        # (1a) advection by mean velocity: 1/rho0*\partialz(rho0 <w><phi>)
        # (1b) turbulent advection: 1/rho0*\partialz(<rho0 w'phi'>)

        cdef:
            Py_ssize_t index
            Py_ssize_t scalar_count=0
            Py_ssize_t k
            Py_ssize_t gw = Gr.gw
            Py_ssize_t w_index = M1.name_index['w']
            # double dzi = Gr.dzi
            double dzi2 = 0.5*Gr.dzi

            double [:,:] M1_values = M1.values
            double [:,:] tendency_M1 = M1.tendencies
            double [:,:] flux = self.flux
            double [:,:] tendency = self.tendencies

            double [:] rho0 = Ref.rho0
            double [:] alpha0 = Ref.alpha0

        for index in xrange(M1.nv): #Loop over the prognostic variables
            if M1.var_type[index] == 1: # Only compute advection if variable i is a scalar
                flux_index = scalar_count    # The flux has a different shift since it is only for the scalars
                # print('scalar count', scalar_count)
                # print('scalar shift', scalar_shift)
                # print('flux_shift', flux_shift)

                for k in xrange(Gr.nzg):
                    flux[flux_index,k] = rho0[k]*M1_values[w_index,k]*M1_values[index,k]
                for k in xrange(gw,Gr.nzg-gw):
                    tendency[flux_index,k] = - alpha0[k]*(flux[flux_index,k+1]-flux[flux_index,k-1])*dzi2
                for k in xrange(Gr.nzg):
                    tendency_M1[index,k] += tendency[flux_index,k]

                scalar_count += 1

        return


    cpdef update_M2_2nd(self, Grid Gr, ReferenceState Ref, SecondOrderMomenta M2):
        return


    # cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return


    def plot(self, Grid Gr, TimeStepping TS, MeanVariables M1):
        if np.isnan(self.tendencies).any():
            print('!!!!! NAN in SA Tendencies M1')
            if np.isnan(self.tendencies[0,:]).any():
                print('!!!!! NAN in SA Tendencies M1, 0')
            if np.isnan(self.tendencies[1,:]).any():
                print('!!!!! NAN in SA Tendencies M1, 1')
            if np.isnan(self.tendencies[2,:]).any():
                print('!!!!! NAN in SA Tendencies M1, 2')
            if np.isnan(self.tendencies[3,:]).any():
                print('!!!!! NAN in SA Tendencies M1, 3')
        if np.isnan(self.flux).any():
            print('!!!!! NAN in SA Fluxes M1')

        print('SA', self.tendencies.shape, self.flux.shape)

        plt.figure(1,figsize=(15,7))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,5,1)
        plt.plot(M1.tendencies[3,:], Gr.z, 'x-', label='tend th')
        plt.title('M1.Tendencies')
        plt.legend(loc=3)
        plt.subplot(1,5,2)
        plt.plot(M1.tendencies[0,:], Gr.z, 'x-', label='tend u')
        plt.plot(M1.tendencies[1,:], Gr.z, 'x-', label='tend v')
        plt.plot(M1.tendencies[2,:], Gr.z, 'x-', label='tend w')
        plt.title('M1.Tendencies')
        plt.legend(loc=3)
        # plt.subplot(1,4,2)
        # plt.plot(self.tendencies[0,:], Gr.z, 'x-', label='tend u')
        # plt.plot(self.tendencies[1,:], Gr.z, 'x-', label='tend v')
        # plt.plot(self.tendencies[2,:], Gr.z, 'x-', label='tend w')
        # plt.title('self.Tendencies')
        # plt.legend(loc=3)
        plt.subplot(1,5,3)
        plt.plot(self.flux[0,:], Gr.z, 'x-', label='flux th')
        plt.title('self.Fluxes')
        plt.legend(loc=4)

        plt.subplot(1,5,4)
        plt.plot(M1.values[3,:], Gr.z, 'x-', label=' th')
        plt.title('M1.values')
        plt.legend(loc=4)
        plt.subplot(1,5,5)
        plt.plot(M1.values[0,:], Gr.z, 'x-', label=' u')
        plt.plot(M1.values[1,:], Gr.z, 'x-', label=' v')
        plt.plot(M1.values[2,:], Gr.z, 'x-', label=' w')
        plt.title('M1.values')
        plt.legend(loc=4)
        # plt.show()
        plt.savefig('./figs/SA_' + np.str(np.int(TS.t)) + '.png')
        plt.close()
        return