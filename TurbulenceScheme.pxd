from Grid cimport Grid
cimport PrognosticVariables
# cimport DiagnosticVariables
# cimport Kinematics
# cimport Surface
# from NetCDFIO cimport NetCDFIO_Stats


cdef class TurbulenceNone:

    # cdef:
    #     double const_viscosity

    # cpdef initialize(self, Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS)
    cpdef initialize(self)
    # cpdef update(self, Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
    #              PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, Surface.SurfaceBase Sur)
    cpdef update(self)
    # cpdef stats_io(self, Grid Gr, DiagnosticVariables.DiagnosticVariables DV,
    #                PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS)
    cpdef stats_io(self)
