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


def SGSFactory(namelist):
    if(namelist['sgs']['scheme'] == 'UniformViscosity'):
        print('SGS scheme:', namelist['sgs']['scheme'])
        return UniformViscosity(namelist)
    else:
        print('SGS scheme:', namelist['sgs']['scheme'])
        return UniformViscosity(namelist)


cdef class UniformViscosity:
    def __init__(self,namelist):
        try:
            self.const_diffusivity = namelist['sgs']['UniformViscosity']['diffusivity']
        except:
            self.const_diffusivity = 0.0

        try:
            self.const_viscosity = namelist['sgs']['UniformViscosity']['viscosity']
        except:
            self.const_viscosity = 0.0

        self.is_init = False
        print('SGS const, DV:', self.const_diffusivity)
        print('SGS const, EV:', self.const_viscosity)

        return


    # cpdef initialize(self, Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef initialize(self):

        return


    # cpdef update(self, Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
    #              PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, Surface.SurfaceBase Sur):
    cpdef update(self):


        # cdef:
            # Py_ssize_t diff_shift = DV.get_varshift(Gr,'diffusivity')
            # Py_ssize_t visc_shift = DV.get_varshift(Gr,'viscosity')
            # Py_ssize_t i
        # with nogil:
        #     if not self.is_init:
        #         for i in xrange(Gr.dims.npg):
        #             DV.values[diff_shift + i] = self.const_diffusivity
        #             DV.values[visc_shift + i] = self.const_viscosity
        #             self.is_init = True
        return


    # cpdef stats_io(self, Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
    #              PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return

