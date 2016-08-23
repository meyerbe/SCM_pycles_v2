import netCDF4 as nc
import numpy as np
cimport numpy as np
from scipy.interpolate import PchipInterpolator,pchip_interpolate

from NetCDFIO cimport NetCDFIO_Stats
from Grid cimport Grid
from ReferenceState cimport ReferenceState
from PrognosticVariables cimport MeanVariables
from PrognosticVariables cimport SecondOrderMomenta

from thermodynamic_functions cimport exner_c, entropy_from_thetas_c, thetas_t_c, qv_star_c, thetas_c
cimport ReferenceState

from libc.math cimport sqrt, fmin, cos, exp, fabs
include 'parameters.pxi'

cdef class InitializationBase:
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref)
    cpdef initialize_profiles(self, Grid Gr, ReferenceState Ref, MeanVariables M1, SecondOrderMomenta M2)
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref )
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, MeanVariables MV)

cdef class InitSoares(InitializationBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref)
    cpdef initialize_profiles(self, Grid Gr, ReferenceState Ref, MeanVariables M1, SecondOrderMomenta M2)
    # cpdef initialize_surface(self, Grid Gr, ReferenceState Ref )
    # cpdef initialize_io(self, NetCDFIO_Stats Stats)
    # cpdef update_surface(self, MeanVariables MV)