from Grid cimport Grid
cimport PrognosticVariables
# cimport DiagnosticVariables
# cimport Kinematics
# cimport Surface
# from NetCDFIO cimport NetCDFIO_Stats

cdef class TurbulenceBase:
    cpdef initialize(self)
    cpdef update(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef update_M1(self,Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef stats_io(self)

cdef class TurbulenceNone(TurbulenceBase):
    cpdef initialize(self)
    # cpdef update(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    # cpdef stats_io(self)


cdef class Turbulence2ndOrder(TurbulenceBase):
    # cdef:
    #     double const_viscosity
    cpdef initialize(self)
    cpdef update(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef advect_M2_local(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef stats_io(self)
