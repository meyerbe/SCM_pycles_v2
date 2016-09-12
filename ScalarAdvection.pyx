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
            Py_ssize_t scalar_index
            Py_ssize_t scalar_count=0
            Py_ssize_t k
            Py_ssize_t w_index = M1.name_index['w']
            # double dzi = Gr.dzi
            double dzi2 = 0.5*Gr.dzi

            double [:,:] M1_values = M1.values
            double [:,:] flux = self.flux
            double [:,:] tendency = self.tendencies
            double [:,:] tendency_M1 = M1.tendencies

            double [:] rho0 = Ref.rho0
            double [:] rho0_half = Ref.rho0_half
            double [:] alpha0 = Ref.alpha0
            double [:] alpha0_half = Ref.alpha0_half

            # Py_ssize_t s_shift = PV.get_varshift(Gr,'s')
            # Py_ssize_t t_shift = DV.get_varshift(Gr,'temperature')
            # Py_ssize_t ql_shift, qv_shift, qt_shift

        for scalar_index in xrange(M1.nv): #Loop over the prognostic variables
            if M1.var_type[scalar_index] == 1: # Only compute advection if variable i is a scalar
                flux_index = scalar_count    # The flux has a different shift since it is only for the scalars
                # print('scalar count', scalar_count)
                # print('scalar shift', scalar_shift)
                # print('flux_shift', flux_shift)

                for k in xrange(1,Gr.nzg-1):
                    # scalar_int = 0.5*(M1_values[scalar_index,k]+M1_values[scalar_index,k+1])
                    # flux[flux_index,k] = rho0[k]*M1_values[w_index,k]*scalar_int
                    # tendency[flux_index,k] = - alpha0_half[k]*(flux[flux_index,k]-flux[flux_index,k-1])*dzi
                    flux[flux_index,k] = rho0[k]*M1_values[w_index,k]*M1_values[scalar_index,k]
                    tendency[flux_index,k] = - alpha0[k]*(flux[flux_index,k+1]-flux[flux_index,k-1])*dzi2
                scalar_count += 1


        # print(tendency.shape, tendency.size, scalar_shift, k)
        # compute_qt_sedimentation_s_source(&Gr.dims, &Rs.p0_half[0],  &Rs.rho0_half[0], &self.flux[flux_shift],
        #                         &PV.values[qt_shift], &DV.values[qv_shift], &DV.values[t_shift],
        #                         &PV.tendencies[s_shift], self.Lambda_fp,self.L_fp, Gr.dims.dx[d],d)
        return


    cpdef update_M2_2nd(self, Grid Gr, ReferenceState Ref, SecondOrderMomenta M2):
        return


    # cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return
