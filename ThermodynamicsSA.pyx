#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

cimport numpy as np
import numpy as np

# cimport Grid
# cimport PrognosticVariables
import cython
from libc.math cimport fmax, fmin

cdef extern from "entropies.h":
    # Specific entropy of dry air
    inline double sd_c(double pd, double T) nogil
    # Specific entropy of water vapor
    inline double sv_c(double pv, double T) nogil
    # Specific entropy of condensed water
    inline double sc_c(double L, double T) nogil



cdef class ThermodynamicsSA:
    def __init__(self, namelist):
    # def __init__(self, dict namelist, LatentHeat LH):
        '''
        Init method saturation adjsutment thermodynamics.
        :param namelist: dictionary
        :param LH: LatentHeat class instance
        :return:
        '''

        # self.L_fp = LH.L_fp
        # self.Lambda_fp = LH.Lambda_fp
        # self.CC = ClausiusClapeyron()
        # self.CC.initialize(namelist, LH, Par)

        #Check to see if qt clipping is to be done. By default qt_clipping is on.
        try:
            self.do_qt_clipping = namelist['thermodynamics']['do_qt_clipping']
        except:
            self.do_qt_clipping = True

        return


    # cpdef initialize(self,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS):
    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):

        # PV.add_variable('s','m/s',"sym","scalar")
        M1.add_variable('s','m/s',"scalar")
        M1.add_variable('qt','g/kg',"scalar")

        return



    # cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
    #                  PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV):
    cpdef update(self):

        return



# cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
#                    DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return

