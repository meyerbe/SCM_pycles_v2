#!python
#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: cdivision=True


import netCDF4 as nc
import numpy as np
cimport numpy as np
from scipy.interpolate import PchipInterpolator,pchip_interpolate

from NetCDFIO cimport NetCDFIO_Stats
from Grid cimport Grid
from ReferenceState cimport ReferenceState
from PrognosticVariables cimport MeanVariables
from PrognosticVariables cimport SecondOrderMomenta
# cimport DiagnosticVariables

from thermodynamic_functions cimport exner_c, entropy_from_thetas_c, thetas_t_c, qv_star_c, thetas_c

from libc.math cimport sqrt, fmin, cos, exp, fabs
include 'parameters.pxi'



def InitializationFactory(namelist):


        casename = namelist['meta']['casename']
        # if casename == 'SullivanPatton':
        #     return InitSullivanPatton
        # elif casename == 'StableBubble':
        #     return InitStableBubble
        # elif casename == 'SaturatedBubble':
        #     return InitSaturatedBubble
        # elif casename == 'Bomex':
        #     return InitBomex
        # elif casename == 'Gabls':
        #     return InitGabls
        # elif casename == 'DYCOMS_RF01':
        #     return InitDYCOMS_RF01
        # elif casename == 'DYCOMS_RF02':
        #     return InitDYCOMS_RF02
        # elif casename == 'SMOKE':
        #     return InitSmoke
        # elif casename == 'Rico':
        #     return InitRico
        # elif casename == 'CGILS':
        #     return  InitCGILS
        # elif casename == 'ZGILS':
        #     return  InitZGILS
        # elif casename == 'DCBLSoares':
        if casename == 'DCBLSoares':
            return InitSoares()
        # elif casename == 'DCBLSoares_moist':
        #     return InitSoares_moist
        else:
            pass



cdef class InitializationBase:
    def __init__(self):
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
        return
    cpdef initialize_profiles(self, Grid Gr, ReferenceState Ref, MeanVariables M1, SecondOrderMomenta M2):
        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        # Stats.add_ts('Tsurface')
        # Stats.add_ts('shf')
        # Stats.add_ts('lhf')
        # Stats.add_ts('ustar')
        return
    cpdef update_surface(self, MeanVariables MV):
        return



# def InitSoares(namelist, Grid Gr,PrognosticVariables.PrognosticVariables PV,
#                        ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS, LatentHeat La):
# def InitSoares(Grid Gr,PrognosticVariables.PrognosticVariables PV,
#                        ReferenceState.ReferenceState RS, Th, NetCDFIO_Stats NS):
cdef class InitSoares(InitializationBase):
# cdef class InitSoares:
    def __init__(self):
        print('Initializing DCBL Soares')
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
        #Generate the reference profiles
        Ref.Pg = 1.0e5      # Pressure at ground (Soares)
        Ref.Tg = 300.0      # Temperature at ground (Soares)
        Ref.qtg = 5e-3      # Total water mixing ratio at surface: qt = 5 g/kg (Soares)
        Ref.u0 = 0.01       # velocities removed in Galilean transformation (Soares: u = 0.01 m/s, IOP: 0.0 m/s)
        Ref.v0 = 0.0        # (Soares: v = 0.0 m/s)

        # Ref.initialize(Gr, Th, NS, Pa)       # initialize reference state; done for every case
        Ref.initialize(Gr, NS)

        return



    # #Get the variable number for each of the velocity components
    # np.random.seed(Pa.rank)
    # cdef:
    #     Py_ssize_t u_varshift = PV.get_varshift(Gr,'u')
    #     Py_ssize_t v_varshift = PV.get_varshift(Gr,'v')
    #     Py_ssize_t w_varshift = PV.get_varshift(Gr,'w')
    #     Py_ssize_t s_varshift = PV.get_varshift(Gr,'s')
    #     # Py_ssize_t qt_varshift = PV.get_varshift(Gr,'qt')       # !!!! Problem: if dry Microphysics scheme chosen: qt is no PV
    #     Py_ssize_t i,j,k
    #     Py_ssize_t ishift, jshift, e_varshift
    #     Py_ssize_t ijk
    #     double [:] theta = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
    #     # double [:] qt = np.empty((Gr.dims.nlg[2]),dtype=np.double,order='c')
    #     double temp
    #
    #     #Generate initial perturbations (here we are generating more than we need)      ??? where amplitude of perturbations given?
    #     cdef double [:] theta_pert = np.random.random_sample(Gr.dims.nzg)
    #     cdef double theta_pert_
    #
    #
    # # Initial theta (potential temperature) profile (Soares)
    # for k in xrange(Gr.dims.nlg[2]):
    #     # if Gr.zl_half[k] <= 1350.0:
    #     #     theta[k] = 300.0
    #     # else:
    #     #     theta[k] = 300.0 + 2.0/1000.0 * (Gr.zl_half[k] - 1350.0)
    #     theta[k] = 297.3 + 2.0/1000.0 * (Gr.zl_half[k])
    #
    #
    # cdef double [:] p0 = RS.p0_half
    #
    # # Now loop and set the initial condition
    # for i in xrange(Gr.dims.nlg[0]):
    #     ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
    #     for j in xrange(Gr.dims.nlg[1]):
    #         jshift = j * Gr.dims.nlg[2]
    #         for k in xrange(Gr.dims.nlg[2]):
    #             ijk = ishift + jshift + k
    #             PV.values[u_varshift + ijk] = 0.0 - RS.u0       # original Soares: u = 0.1
    #             PV.values[v_varshift + ijk] = 0.0 - RS.v0
    #             PV.values[w_varshift + ijk] = 0.0
    #
    #             # Set the entropy prognostic variable including a potential temperature perturbation
    #             # fluctuation height = 200m; fluctuation amplitude = 0.1 K
    #             if Gr.zl_half[k] < 200.0:
    #                 theta_pert_ = (theta_pert[ijk] - 0.5)* 0.1
    #             else:
    #                 theta_pert_ = 0.0
    #             temp = (theta[k] + theta_pert_)*exner_c(RS.p0_half[k])
    #             PV.values[s_varshift + ijk] = Th.entropy(RS.p0_half[k],temp,0.0,0.0,0.0)
    #
    # if 'e' in PV.name_index:
    #     e_varshift = PV.get_varshift(Gr, 'e')
    #     for i in xrange(Gr.dims.nlg[0]):
    #         ishift =  i * Gr.dims.nlg[1] * Gr.dims.nlg[2]
    #         for j in xrange(Gr.dims.nlg[1]):
    #             jshift = j * Gr.dims.nlg[2]
    #             for k in xrange(Gr.dims.nlg[2]):
    #                 ijk = ishift + jshift + k
    #                 PV.values[e_varshift + ijk] = 0.0

    # return



