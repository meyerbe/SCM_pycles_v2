#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport PrognosticVariables
cimport ReferenceState
# cimport DiagnosticVariables
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats

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

    # cpdef initialize(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef initialize(self):
        # self.flux = np.zeros((PV.nv_scalars*Gr.dims.npg*Gr.dims,),dtype=np.double,order='c')

        return


    # cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs,PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
    cpdef update(self):

        return


    # cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return
