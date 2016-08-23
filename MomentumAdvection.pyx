#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport Grid
cimport PrognosticVariables
cimport ReferenceState
from NetCDFIO cimport NetCDFIO_Stats

import numpy as np
cimport numpy as np
import sys

cdef class MomentumAdvection:
    def __init__(self, namelist):
        try:
            self.order = namelist['momentum_transport']['order']
            print('momentum transport: order: ' + np.str(self.order))
        except:
            print('momentum_transport order not given in namelist')
            print('Killing simulation now!')
            sys.exit()

        return

    # cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef initialize(self):

        return

    # cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV):
    cpdef update(self):

        return


    # cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return
