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

from thermodynamic_functions import entropy_from_tp
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
        if casename == 'Bomex':
            return InitBomex()
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
        elif casename == 'DCBLSoares':
            return InitSoares()
        # elif casename == 'DCBLSoares_moist':
        #     return InitSoares_moist
        elif casename == 'Test':
            return InitTest()
        else:
            pass



cdef class InitializationBase:
    def __init__(self):
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
        return
    cpdef initialize_profiles(self, Grid Gr, ReferenceState Ref, MeanVariables M1, SecondOrderMomenta M2, NetCDFIO_Stats NS):
        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
    # cpdef initialize_surface(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
    #     self.u_flux = 0.0
    #     self.v_flux = 0.0
    #     self.qt_flux = 0.0
    #     self.s_flux = 0.0
    #
    #     self.obukhov_length = 0.0
    #     self.friction_velocity = 0.0
    #     self.shf = 0.0
    #     self.lhf = 0.0
    #     self.b_flux = 0.0
    #
    #     NS.add_ts('uw_surface_mean', Gr)
    #     NS.add_ts('vw_surface_mean', Gr)
    #     NS.add_ts('s_flux_surface_mean', Gr)
    #     NS.add_ts('shf_surface_mean', Gr)
    #     NS.add_ts('lhf_surface_mean', Gr)
    #     NS.add_ts('obukhov_length_mean', Gr)
    #     NS.add_ts('friction_velocity_mean', Gr)
    #     NS.add_ts('buoyancy_flux_surface_mean', Gr)
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        # Stats.add_ts('Tsurface')
        # Stats.add_ts('shf')
        # Stats.add_ts('lhf')
        # Stats.add_ts('ustar')
        return
    cpdef update_surface(self, MeanVariables MV):
        return

    # cpdef initialize_entropy(self, double [:] theta, Grid Gr, ReferenceState Ref, MeanVariables M1):
    #     cdef:
    #         double temp
    #         Py_ssize_t k
    #         Py_ssize_t s_varshift = M1.get_varshift(Gr,'s')
    #         double min = self.pert_min
    #         double max = self.pert_max
    #
    #     cdef double [:] theta_pert = np.random.random_sample(Gr.nzg)
    #     cdef double theta_pert_
    #
    #     for k in xrange(Gr.nzg):
    #         # M1.values[s_varshift + k] = Th.entropy(Ref.p0_half[k],temp,0.0,0.0,0.0)
    #         if Gr.z_half[k] < max:
    #             theta_pert_ = (theta_pert[k] - 0.5)* 0.1
    #         else:
    #             theta_pert_ = 0.0
    #             temp = (theta[k] + theta_pert_)*exner_c(Ref.p0_half[k])
    #         M1.values[s_varshift + k] = entropy_from_tp(Ref.p0_half[k],temp,0.0,0.0,0.0)
    #     return







cdef class InitSoares(InitializationBase):
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

        Ref.initialize(Gr, NS)

        return


    cpdef initialize_profiles(self, Grid Gr, ReferenceState Ref, MeanVariables M1, SecondOrderMomenta M2, NetCDFIO_Stats NS):

        # (1) Generate initial perturbations
        self.pert_min = 0.0
        self.pert_max = 200.0
        cdef double [:] theta_pert = np.random.random_sample(Gr.nzg)
        cdef double theta_pert_

        # (2) Initialize Mean Variables
        # np.random.seed(Pa.rank)
        # print(M1.name_index.keys())
        cdef:
            Py_ssize_t u_varshift = M1.get_varshift(Gr,'u')
            Py_ssize_t v_varshift = M1.get_varshift(Gr,'v')
            Py_ssize_t w_varshift = M1.get_varshift(Gr,'w')
            Py_ssize_t th_varshift = M1.get_varshift(Gr,'th')
            Py_ssize_t k
            # Py_ssize_t e_varshift
            double [:] theta = np.empty((Gr.nzg),dtype=np.double,order='c')
            double temp
            # double [:] p0 = Ref.p0_half

        # (i) Theta (potential temperature) profile (Soares) incl. perturbations
        # fluctuation height = 200m; fluctuation amplitude = 0.1 K
        for k in xrange(Gr.nzg):
            if Gr.z_half[k] <= 1350.0:
                theta[k] = 300.0
            else:
                # theta[k] = 300.0 + 2.0/1000.0 * (Gr.z_half[k] - 1350.0)
               theta[k] = 297.3 + 2.0/1000.0 * (Gr.z_half[k])


        # # (ii) Velocities & Entropy
        cdef:
            double qt = 0.0
            double ql = 0.0
            double qi = 0.0
        print('Initializing Velocity and Entropy')
        for k in xrange(Gr.nzg):
            if Gr.z_half[k] < 200.0:
                theta_pert_ = (theta_pert[k] - 0.5)* 0.1
            else:
                theta_pert_ = 0.0
            temp = (theta[k] + theta_pert_)*exner_c(Ref.p0_half[k])
            M1.values[th_varshift + k] = entropy_from_tp(Ref.p0_half[k],temp,qt,ql,qi)               # s = Thermodynamics.entropy(p_half[k],temperature_half[k],self.qtg,ql_half[k],qi_half[k])
            M1.values[u_varshift + k] = 0.0
            M1.values[v_varshift + k] = 0.0
            M1.values[w_varshift + k] = 0.0

        # (2) Initialize Second Order Momenta
        print(M2.name_index.keys())
        cdef:
            Py_ssize_t ww_varshift = M2.get_varshift(Gr,'ww')
        for k in xrange(Gr.nzg):
            M2.values[ww_varshift + k] = 0.0


         # # if 'e' in PV.name_index:
        # #     e_varshift = PV.get_varshift(Gr, 'e')
        # #     for k in xrange(Gr.nzg):
        # #       PV.values[e_varshift + k] = 0.0

        return



cdef class InitBomex(InitializationBase):
# cdef class InitSoares:
    def __init__(self):
        print('Initializing Bomex')
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
        #Generate the reference profiles
        Ref.Pg = 0.0
        Ref.Tg = 0.0
        Ref.qtg = 0.0
        Ref.u0 = 0.0
        Ref.v0 = 0.0

        Ref.initialize(Gr, NS)

        return


    cpdef initialize_profiles(self, Grid Gr, ReferenceState Ref, MeanVariables M1, SecondOrderMomenta M2, NetCDFIO_Stats NS):


        return




cdef class InitTest(InitializationBase):
    def __init__(self):
        print('Initializing Test')
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
        #Generate the reference profiles
        Ref.Pg = 1.0e5      # Pressure at ground (Soares)
        Ref.Tg = 300.0      # Temperature at ground (Soares)
        Ref.qtg = 5e-3      # Total water mixing ratio at surface: qt = 5 g/kg (Soares)
        Ref.u0 = 0.01       # velocities removed in Galilean transformation (Soares: u = 0.01 m/s, IOP: 0.0 m/s)
        Ref.v0 = 0.0        # (Soares: v = 0.0 m/s)

        Ref.initialize(Gr, NS)
        return


    cpdef initialize_profiles(self, Grid Gr, ReferenceState Ref, MeanVariables M1, SecondOrderMomenta M2, NetCDFIO_Stats NS):

        # (1) Generate initial perturbations
        # self.pert_min = 0.0
        # self.pert_max = 200.0
        # cdef double [:] theta_pert = np.random.random_sample(Gr.nzg)
        # cdef double theta_pert_

        # (2) Initialize Mean Variables
        cdef:
            Py_ssize_t u_varshift = M1.get_varshift(Gr,'u')
            Py_ssize_t v_varshift = M1.get_varshift(Gr,'v')
            Py_ssize_t w_varshift = M1.get_varshift(Gr,'w')
            Py_ssize_t th_varshift = M1.get_varshift(Gr,'th')
            Py_ssize_t k
            Py_ssize_t nv_vel = M1.nv_velocities

            # double [:] s = M1.values[s_varshift:s_varshift+Gr.nzg]

            # double [:] p0 = Ref.p0_half

        # (i) Theta (potential temperature) profile (Soares) incl. perturbations
        for k in xrange(Gr.nzg):
            M1.values[th_varshift+k] = 6000.0
            M1.values[u_varshift+k] = 0.0
            M1.values[v_varshift+k] = 0.0
            M1.values[w_varshift+k] = 0.0

        # # (ii) Velocities & Entropy
        cdef:
            double qt = 0.0
            double ql = 0.0
            double qi = 0.0
        print('Initializing Velocity and Entropy')

        # (2) Initialize Second Order Momenta

        return