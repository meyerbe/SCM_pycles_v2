#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from Grid cimport Grid
cimport PrognosticVariables
from TimeStepping cimport TimeStepping
# cimport DiagnosticVariables
# cimport Kinematics
# cimport Surface
# from NetCDFIO cimport NetCDFIO_Stats
from libc.math cimport exp, sqrt
cimport numpy as np
import numpy as np
import pylab as plt
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


    def initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('initializing UniformViscosity')
        self.is_init = False
        self.viscosity_M1 = np.zeros((M1.nv_velocities,Gr.nzg),dtype=np.double,order='c')
        self.diffusivity_M1 = np.zeros((M1.nv_scalars,Gr.nzg),dtype=np.double,order='c')
        self.viscosity_M2 = np.zeros((M2.nv_velocities,Gr.nzg),dtype=np.double,order='c')
        self.diffusivity_M2 = np.zeros((M2.nv_scalars,Gr.nzg),dtype=np.double,order='c')
        return


    def update(self, Grid Gr):
        cdef Py_ssize_t k

        if not self.is_init:
            self.is_init = True
            for k in xrange(Gr.nzg):
                self.viscosity_M1[:,k] = self.const_viscosity
            for k in xrange(Gr.nzg):
                self.viscosity_M2[:,k] = self.const_viscosity
            for k in xrange(Gr.nzg):
                self.diffusivity_M1[:,k] = self.const_diffusivity
            for k in xrange(Gr.nzg):
                self.diffusivity_M2[:,k] = self.const_diffusivity
        return


    # cpdef stats_io(self, Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
    #              PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS):
    def stats_io(self):

        return

    def plot(self, Grid Gr, TimeStepping TS):
        if np.isnan(self.viscosity_M1).any():
            print('!!!!! NAN in viscosity M1')
        if np.isnan(self.diffusivity_M1).any():
            print('!!!!! NAN in diffusivity M1')
        if np.isnan(self.viscosity_M2).any():
            print('!!!!! NAN in viscosity M2')
        if np.isnan(self.diffusivity_M2).any():
            print('!!!!! NAN in diffusivity M2')


        plt.figure(1,figsize=(12,6))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(self.viscosity_M1[0,:], Gr.z)
        plt.plot(self.viscosity_M1[1,:], Gr.z)
        plt.plot(self.viscosity_M1[2,:], Gr.z)
        plt.title('M1 viscosity')
        plt.subplot(1,4,2)
        plt.plot(self.diffusivity_M1[0,:], Gr.z)
        plt.title('M1 diffusivity')
        plt.subplot(1,4,3)
        plt.plot(self.viscosity_M2[0,:], Gr.z)
        plt.title('M2 viscosity')
        plt.subplot(1,4,4)
        plt.plot(self.diffusivity_M2[0,:], Gr.z)
        plt.title('M2 diffusivity')
        # plt.show()
        # plt.savefig('./figs/diffusivity_' + message + '_' + np.str(TS.t) + '.png')
        plt.savefig('./figs/diffusivity_' + np.str(TS.t) + '.png')
        plt.close()
        return



def SGSFactory(namelist):
    if(namelist['sgs']['scheme'] == 'UniformViscosity'):
        print('SGS scheme:', namelist['sgs']['scheme'])
        return UniformViscosity(namelist)
    else:
        print('SGS scheme:', namelist['sgs']['scheme'])
        return UniformViscosity(namelist)
