from Grid cimport Grid
from ReferenceState cimport ReferenceState
cimport PrognosticVariables
from NetCDFIO cimport NetCDFIO_Stats

cdef class Diffusion:
    cdef:
        double [:,:] flux_M1
        double [:,:] tendencies_M1
        # Py_ssize_t n_fluxes
        double [:,:] grad

    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1)
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, SGS)
    cpdef stats_io(self)


    cpdef update_M1(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, SGS)