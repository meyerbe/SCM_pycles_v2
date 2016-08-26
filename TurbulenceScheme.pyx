#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from Grid cimport Grid
cimport PrognosticVariables
# cimport DiagnosticVariables
# cimport Kinematics
# cimport Surface
# from NetCDFIO cimport NetCDFIO_Stats
from libc.math cimport exp, sqrt
cimport numpy as np
import numpy as np
import cython


def TurbulenceFactory(namelist):
    if(namelist['turbulence']['scheme'] == '2nd order'):
        print('Turbulence scheme:', namelist['turbulence']['scheme'])
        return TurbulenceNone(namelist)
    else:
        print('Turbulence scheme not given.')
        return TurbulenceNone()


cdef class TurbulenceNone:
    def __init__(self,namelist):

        return

    # cpdef initialize(self, Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef initialize(self):

        return


    # cpdef update(self, Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
    #              PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, Surface.SurfaceBase Sur):
    cpdef update(self):
        # (1) Advection: MA.update_M2, SA.update_M2
        # (2) Diffusion: SD.update_M2, MD.update_M2
        # (3) Pressure: ?

        # cdef:
            # Py_ssize_t diff_shift = DV.get_varshift(Gr,'diffusivity')
            # Py_ssize_t visc_shift = DV.get_varshift(Gr,'viscosity')
            # Py_ssize_t i
        # with nogil:
        #     if not self.is_init:
        #         pass



        return


    # cpdef stats_io(self, Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
    #              PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return

