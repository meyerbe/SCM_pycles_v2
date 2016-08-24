#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport numpy as np
import numpy as np

cimport Grid
cimport PrognosticVariables
cimport Thermodynamics
from thermodynamic_functions cimport thetas_c
import cython

cdef extern from "entropies.h":
    inline double sd_c(double p0, double T) nogil

cdef class ThermodynamicsDry:
    # def __init__(self,namelist,LatentHeat LH):
    def __init__(self,namelist):
        # self.L_fp = LH.L_fp
        # self.Lambda_fp = LH.Lambda_fp
        # self.CC = ClausiusClapeyron()
        # self.CC.initialize(namelist,LH,Pa)


        return

    # cpdef initialize(self,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS):
    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        # PV.add_variable('s','m/s',"sym","scalar")
        M1.add_variable('s','m/s',"scalar")

        return



    # cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
    #                  PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
    cpdef update(self):

        return



# cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
#                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return