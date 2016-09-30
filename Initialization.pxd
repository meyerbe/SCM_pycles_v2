import netCDF4 as nc
import numpy as np
cimport numpy as np
from scipy.interpolate import PchipInterpolator,pchip_interpolate

from NetCDFIO cimport NetCDFIO_Stats
from Grid cimport Grid
from ReferenceState cimport ReferenceState
from TimeStepping cimport TimeStepping
from PrognosticVariables cimport MeanVariables
from PrognosticVariables cimport SecondOrderMomenta

from thermodynamic_functions cimport exner, entropy_from_thetas_c, thetas_t_c, qv_star_c, thetas_c
cimport ReferenceState

from libc.math cimport sqrt, fmin, cos, exp, fabs
include 'parameters.pxi'

cdef class InitializationBase:
    cdef:
        double pert_min
        double pert_max
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS)
    cpdef initialize_profiles(self, Grid Gr, ReferenceState Ref, TimeStepping TS, MeanVariables M1, SecondOrderMomenta M2, NetCDFIO_Stats NS)
    #cpdef initialize_surface(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS)
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, MeanVariables MV)
    # cpdef initialize_entropy(self, double [:] theta, Grid Gr, ReferenceState Ref, MeanVariables MV)


cdef class InitSoares(InitializationBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS)
    cpdef initialize_profiles(self, Grid Gr, ReferenceState Ref, TimeStepping TS, MeanVariables M1, SecondOrderMomenta M2, NetCDFIO_Stats NS)
    # cpdef initialize_surface(self, Grid Gr, ReferenceState Ref )
    # cpdef initialize_io(self, NetCDFIO_Stats Stats)
    # cpdef update_surface(self, MeanVariables MV)


cdef class InitTest(InitializationBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS)
    cpdef initialize_profiles(self, Grid Gr, ReferenceState Ref, TimeStepping TS, MeanVariables M1, SecondOrderMomenta M2, NetCDFIO_Stats NS)
    # cpdef initialize_surface(self, Grid Gr, ReferenceState Ref )
    # cpdef initialize_io(self, NetCDFIO_Stats Stats)
    # cpdef update_surface(self, MeanVariables MV)