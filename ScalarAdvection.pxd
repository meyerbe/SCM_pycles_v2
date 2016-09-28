from Grid cimport Grid
from PrognosticVariables cimport MeanVariables
from PrognosticVariables cimport SecondOrderMomenta
from ReferenceState cimport ReferenceState
from TimeStepping cimport TimeStepping
# # cimport DiagnosticVariables
# cimport TimeStepping
# from NetCDFIO cimport NetCDFIO_Stats
# # from Thermodynamics cimport LatentHeat

cdef class ScalarAdvection:

    cdef:
        double [:,:] flux
        double [:,:] tendencies
        Py_ssize_t order
        # Py_ssize_t order_sedimentation

    # cpdef initialize(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS)
    cpdef initialize(self, Grid Gr, MeanVariables M1)
    # cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs,PrognosticVariables.PrognosticVariables PV,  DiagnosticVariables.DiagnosticVariables DV)
    cpdef update(self, Grid Gr, ReferenceState Ref, MeanVariables M1)
    cpdef update_M1_2nd(self, Grid Gr, ReferenceState Ref, MeanVariables M1)
    cpdef update_M2_2nd(self, Grid Gr, ReferenceState Ref, SecondOrderMomenta M2)
    # cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS)
    cpdef stats_io(self)