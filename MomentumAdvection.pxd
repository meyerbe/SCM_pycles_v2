from Grid cimport Grid
from PrognosticVariables cimport MeanVariables
from PrognosticVariables cimport SecondOrderMomenta
from ReferenceState cimport ReferenceState
# from NetCDFIO cimport NetCDFIO_Stats

cdef class MomentumAdvection:
    cdef:
        Py_ssize_t order
        double [:,:] flux
        double [:,:] tendencies

    # cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS)
    cpdef initialize(self, Grid Gr, MeanVariables M1)
    # cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV)
    cpdef update(self, Grid Gr, ReferenceState Rs, MeanVariables M1)
    cpdef update_M1_2nd(self, Grid Gr, ReferenceState Rs, MeanVariables M1)
    cpdef update_M1_4th(self, Grid Gr, ReferenceState Rs, MeanVariables M1)
    cpdef update_M2_2nd(self, Grid Gr, ReferenceState Ref, SecondOrderMomenta M2)
    # cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS)
    cpdef stats_io(self)

    cdef inline double vel_interp_4(self, double phim2, double phim1, double phip1, double phip2) nogil:
    # cdef inline double vel_interp_4(self, double phim2, double phim1, double phip1, double phip2):
        cdef double vel = 7.0/12.0 * (phim1 + phip1) - 1.0/12.0 * (phim2 + phip2 )
        return vel
        # return 7.0/12.0 * (phim1 + phip1) - 1.0/12.0 * (phim2 + phip2 )



