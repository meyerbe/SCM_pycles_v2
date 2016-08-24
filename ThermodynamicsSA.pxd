cimport Grid
cimport PrognosticVariables

cdef class ThermodynamicsSA:
    cdef:
        bint do_qt_clipping
        # double (*L_fp)(double T, double Lambda) nogil
        # double (*Lambda_fp)(double T) nogil
        # ClausiusClapeyron CC


    # cpdef initialize(self,Grid.Grid Gr,PrognosticVariables.PrognosticVariables PV,
    #                  DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2)
    # cpdef entropy(self, double p0, double T, double qt, double ql, double qi)
    # cpdef alpha(self, double p0, double T, double qt, double qv)
    # cpdef eos(self, double p0, double s, double qt)
    # cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
    #           PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV)
    cpdef update(self)
    # cpdef get_pv_star(self, t)
    # cpdef get_lh(self,t)
    # cpdef write_fields(self, Grid.Grid Gr, ReferenceState.ReferenceState RS,
    #              PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
    #                    NetCDFIO_Fields NF, ParallelMPI.ParallelMPI Pa)
    # cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
    #                DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self)
    # cpdef liquid_stats(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,
    #                 DiagnosticVariables.DiagnosticVariables DV, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    #
    # # __
    # cpdef debug_tend(self,message, PrognosticVariables.PrognosticVariables PV_,
    #                  DiagnosticVariables.DiagnosticVariables DV_,
    #                  Grid.Grid Gr_)
