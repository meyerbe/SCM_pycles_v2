cimport Grid
cimport PrognosticVariables
cimport Thermodynamics


cdef class ThermodynamicsDry:
    # cpdef initialize(self,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
    #                  DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS)
    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    # cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
    #              PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV)
    cpdef update(self)
    # cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
    #                DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS)
    cpdef stats_io(self)