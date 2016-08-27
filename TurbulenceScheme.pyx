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

'''
    (0) update M1.tendencies by adding M2.values

    (1) for 2nd (or higher order scheme): update M2.tendencies
        (i) Advection: MA.update_M2, SA.update_M2
        (ii) Diffusion: SD.update_M2, MD.update_M2
        (iii) Pressure: ?
        (iv) Third and higher order terms (first guess set to zero)
'''



def TurbulenceFactory(namelist):
    if(namelist['turbulence']['scheme'] == '2nd_order'):
        print('Turbulence scheme:', namelist['turbulence']['scheme'])
        return Turbulence2ndOrder(namelist)
    elif(namelist['turbulence']['scheme'] == 'None'):
        print('Turbulence scheme:', namelist['turbulence']['scheme'])
        return TurbulenceNone(namelist)
    else:
        print('Turbulence scheme not given.')
        return TurbulenceNone(namelist)


cdef class TurbulenceNone(TurbulenceBase):
    def __init__(self,namelist):
        return
    cpdef initialize(self):
        return
    # cpdef update(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
    #     print('Turb None: update')
    #     return
    # cpdef stats_io(self):
    #     return


cdef class TurbulenceBase:
    def __init__(self,namelist):
        return

    cpdef initialize(self):
        return

    cpdef update(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turbulence Base: update')
        return

    cpdef update_M1(self,Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turb: update M1')
        cdef:
            Py_ssize_t k

            Py_ssize_t u_varshift = M1.get_varshift(Gr, 'u')
            Py_ssize_t v_varshift = M1.get_varshift(Gr, 'v')
            Py_ssize_t w_varshift = M1.get_varshift(Gr, 'w')
            Py_ssize_t s_varshift = M1.get_varshift(Gr, 's')

            Py_ssize_t wu_shift = M2.get_varshift(Gr, 'wu')
            Py_ssize_t wv_shift = M2.get_varshift(Gr, 'wv')
            Py_ssize_t ww_shift = M2.get_varshift(Gr, 'ww')
            Py_ssize_t ws_shift = M2.get_varshift(Gr, 'ws')

        if 'qt' in M1.name_index:
            qt_varshift = M1.get_varshift(Gr, 'qt')
            wqt_shift = M2.get_varshift(Gr, 'wqt')
            for k in xrange(Gr.nzg):
                M1.tendencies[u_varshift + k] +=  M2.values[wu_shift + k]

        for k in xrange(Gr.nzg):
            M1.tendencies[u_varshift + k] -=  M2.values[wu_shift + k]
            M1.tendencies[v_varshift + k] -=  M2.values[wv_shift+ k]
            M1.tendencies[w_varshift+ k] -=  M2.values[ww_shift + k]
            M1.tendencies[s_varshift + k] -=  M2.values[ws_shift + k]
        return

    cpdef stats_io(self):
        return



cdef class Turbulence2ndOrder(TurbulenceBase):
    # (1) Advection: MA.update_M2, SA.update_M2
    # (2) Diffusion: SD.update_M2, MD.update_M2
    # (3) Pressure: ?
    # (4) Third and higher order terms (first guess set to zero)
    def __init__(self,namelist):
        print('initializing Turbulence 2nd')
        return

    # cpdef initialize(self, Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef initialize(self):

        return


    cpdef update(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turbulence 2nd order: update')
        # (1) Advection: MA.update_M2, SA.update_M2
        # (2) Diffusion: SD.update_M2, MD.update_M2
        # (3) Pressure: ?
        # (4) Third and higher order terms (first guess set to zero)
        self.advect_M2_local(Gr, M1, M2)

        return



    cpdef advect_M2_local(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turb update: advect M2 local')
        # implemented for staggered grid
        # w: on w-grid
        # u,v,{s,qt}: on phi-grid
        # —> dz ws, dz wqt on phi-grid      —> ws, wqt on w-grid   -> compare to scalar advection for gradients
        # —> dz wu, dz wv on phi-grid       —> wu, wv on w-grid    -> compare to scalar advection for gradients
        # —> dz ww on w-grid                —> ww on phi-grid      -> compare to momentum advection for gradients

        cdef:
            Py_ssize_t u_varshift = M1.get_varshift(Gr, 'u')
            Py_ssize_t v_varshift = M1.get_varshift(Gr, 'v')
            Py_ssize_t w_varshift = M1.get_varshift(Gr, 'w')
            Py_ssize_t s_varshift = M1.get_varshift(Gr, 's')

            Py_ssize_t wu_shift = M2.get_varshift(Gr, 'wu')
            Py_ssize_t wv_shift = M2.get_varshift(Gr, 'wv')
            Py_ssize_t ww_shift = M2.get_varshift(Gr, 'ww')
            Py_ssize_t ws_shift = M2.get_varshift(Gr, 'ws')

            double [:] u = M1.values[u_varshift:u_varshift+Gr.nzg]
            double [:] v = M1.values[v_varshift:v_varshift+Gr.nzg]
            double [:] w = M1.values[w_varshift:w_varshift+Gr.nzg]
            double [:] s = M1.values[s_varshift:s_varshift+Gr.nzg]

            double dzi = Gr.dzi
            Py_ssize_t k, var_shift

        print('M2: name index', M2.name_index)

        # (i) advection by mean vertical velocity
        for name in M2.name_index:
            if name != 'ww':
            # w and M2.value both on w-grid --> compare to scalar advection for gradients
                for k in xrange(Gr.nzg):
                    var_shift = M2.get_varshift(Gr, name)
                    M2.tendencies[var_shift + k] -= 0.5*(w[k]+w[k+1])*(M2.values[var_shift+k]-M2.values[var_shift+k-1])*dzi
            else:
            # w on w-grid, M2.value on phi-grid --> compare to momentum advection for gradients
                for k in xrange(Gr.nzg):
                    var_shift = M2.get_varshift(Gr, name)
                    M2.tendencies[var_shift + k] -= w[k]*(M2.values[var_shift+k]-M2.values[var_shift+k-1])*dzi

        # (ii) advection by M2
        # w: on w-grid
        # u,v,{s,qt}: on phi-grid
        # —> dz ws, dz wqt on phi-grid      —> ws, wqt on w-grid   -> compare to scalar advection for gradients
        # —> dz wu, dz wv on phi-grid       —> wu, wv on w-grid    -> compare to scalar advection for gradients
        # —> dz ww on w-grid                —> ww on phi-grid      -> compare to momentum advection for gradients

        # wu on w-grid --> interpolate ww; u_mean ok;  wu ok; interpolate w_mean
        # wv on w-grid --> interpolate ww; v_mean ok; wv ok; interpolate w_mean
        # ww on phi-grid --> ww ok; w ok
        # ws on w-grid --> interpolate ww; s_mean ok; ws ok; interpolate w_mean

        for k in xrange(Gr.nzg):
            M2.tendencies[wu_shift+k] -= 0.5*(M2.values[ww_shift+k]+M2.values[ww_shift+k+1])*(u[k]-u[k-1])*dzi \
                                         - M2.values[wu_shift+k]*( 0.5*(w[k]+w[k+1]) - 0.5*(w[k-1]+w[k]) )*dzi
            M2.tendencies[wv_shift+k] -= 0.5*(M2.values[ww_shift+k]+M2.values[ww_shift+k+1])*(v[k]-v[k-1])*dzi \
                                         - M2.values[wv_shift+k]*( 0.5*(w[k]+w[k+1]) - 0.5*(w[k-1]+w[k]) )*dzi
            M2.tendencies[ww_shift+k] -= M2.values[ww_shift+k]*(w[k]-w[k-1])*dzi
            M2.tendencies[ws_shift+k] -= 0.5*(M2.values[ww_shift+k]+M2.values[ww_shift+k+1])*(s[k]-s[k-1])*dzi \
                                         - M2.values[ws_shift+k]*( 0.5*(w[k]+w[k+1]) - 0.5*(w[k-1]+w[k]) )*dzi

        return

    # cpdef stats_io(self, Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
    #              PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return

