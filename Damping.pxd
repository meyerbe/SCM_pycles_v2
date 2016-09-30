from Grid cimport Grid
from ReferenceState cimport ReferenceState
# cimport DiagnosticVariables
# cimport PrognosticVariables as PrognosticVariables
from PrognosticVariables cimport MeanVariables
from PrognosticVariables cimport SecondOrderMomenta



cdef class Damping:
    cdef:
        object scheme
    cpdef initialize(self, Grid Gr, ReferenceState RS)
    # cpdef update(self, Grid Gr, ReferenceState RS, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV)
    cpdef update(self, Grid Gr, ReferenceState RS, MeanVariables M1, SecondOrderMomenta M2)

cdef class Dummy:
    cpdef initialize(self, Grid Gr, ReferenceState RS)
    # cpdef update(self, Grid Gr, ReferenceState RS, PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV)
    cpdef update(self, Grid Gr, ReferenceState RS, MeanVariables M1, SecondOrderMomenta M2)

cdef class Rayleigh:
#     cdef:
#         double z_d  # Depth of damping layer
#         double gamma_r  # Inverse damping timescale
#         double[:] gamma_zhalf
#         double[:] gamma_z
    cpdef initialize(self, Grid Gr, ReferenceState RS)
    # cpdef update(self, Grid Gr, ReferenceState RS, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV)
    cpdef update(self, Grid Gr, ReferenceState RS, MeanVariables M1, SecondOrderMomenta M2)

# cdef class DampingToDomainMean:
#     cdef:
#         double z_d  # Depth of damping layer
#         double gamma_r  # Inverse damping timescale
#         double[:] gamma_zhalf
#         double[:] gamma_z
#     cpdef initialize(self, Grid Gr, ReferenceState RS)
#     # cpdef update(self, Grid Gr, ReferenceState RS, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV)
#     cpdef update(self, Grid Gr, ReferenceState RS, MeanVariables M1, SecondOrderMomenta M2)

# cdef class DampingToDomainMean:
#     cdef:
#         self.damping_depth = None
#         self.damping_timescale = None
#         self.do_damping = None
#         self.damping_zmin = None
#         self.damping_coefficient_center = None
#         self.damping_coefficient_interface = None
#     cpdef initialize(self,case_dict,grid):