from Grid cimport Grid
cimport PrognosticVariables
from ReferenceState cimport ReferenceState
# cimport DiagnosticVariables
# cimport Kinematics
# cimport Surface
# from NetCDFIO cimport NetCDFIO_Stats

cdef class TurbulenceBase:
    cdef double [:,:] buoyancy
    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1)
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef update_M1(self,Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef stats_io(self)

cdef class TurbulenceNone(TurbulenceBase):
    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1)
    # cpdef update(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    # cpdef stats_io(self)


cdef class Turbulence2ndOrder(TurbulenceBase):
    # cdef:
    #     double const_viscosity
    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1)
    cpdef update_M2(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef advect_M2_local(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef pressure_correlations_Mironov(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef pressure_correlations_Andre(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef pressure_correlations_Cheng(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef buoyancy_update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    cpdef stats_io(self)
