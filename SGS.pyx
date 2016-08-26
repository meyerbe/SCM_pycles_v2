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




class UniformViscosity():
    def __init__(self,namelist):
        try:
            self.const_diffusivity = namelist['sgs']['UniformViscosity']['diffusivity']
        except:
            self.const_diffusivity = 0.0
        try:
            self.const_viscosity = namelist['sgs']['UniformViscosity']['viscosity']
        except:
            self.const_viscosity = 0.0

        print('SGS const, DV:', self.const_diffusivity)
        print('SGS const, EV:', self.const_viscosity)

        self.viscosity_M1 = None
        self.diffusivity_M1 = None
        self.viscosity_M2 = None
        self.diffusivity_M2 = None
        return


    # cpdef initialize(self, Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    def initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('initializing UniformViscosity')
        self.is_init = False
        self.viscosity_M1 = np.zeros((M1.nv_velocities*Gr.nzg),dtype=np.double,order='c')
        self.diffusivity_M1 = np.zeros((M1.nv_scalars*Gr.nzg),dtype=np.double,order='c')
        self.viscosity_M2 = np.zeros((M2.nv_velocities*Gr.nzg),dtype=np.double,order='c')
        self.diffusivity_M2 = np.zeros((M2.nv_scalars*Gr.nzg),dtype=np.double,order='c')
        return


    # cpdef update(self, Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
    #              PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, Surface.SurfaceBase Sur):
    def update(self, Grid Gr):
        # Py_ssize_t k
        size_visc_M1 = self.viscosity_M1.size
        size_diff_M1 = self.diffusivity_M1.size
        size_visc_M2 = self.viscosity_M2.size
        size_diff_M2 = self.diffusivity_M2.size

        if not self.is_init:
            self.is_init = True
            for k in xrange(size_visc_M1):
                self.viscosity_M1[k] = self.const_viscosity
            for k in xrange(size_visc_M2):
                self.viscosity_M2[k] = self.const_viscosity
            for k in xrange(size_diff_M1):
                self.diffusivity_M1[k] = self.const_diffusivity
            for k in xrange(size_diff_M2):
                self.diffusivity_M2[k] = self.const_diffusivity
        return


    # cpdef stats_io(self, Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
    #              PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS):
    def stats_io(self):

        return



def SGSFactory(namelist):
    if(namelist['sgs']['scheme'] == 'UniformViscosity'):
        print('SGS scheme:', namelist['sgs']['scheme'])
        return UniformViscosity(namelist)
    else:
        print('SGS scheme:', namelist['sgs']['scheme'])
        return UniformViscosity(namelist)
