#!python
# cython: boundscheck=False
# cython: wraparound=True
# cython: initializedcheck=False
# cython: cdivision=True

cimport Grid
# cimport Restart
cimport numpy as np
import numpy as np
import sys

from NetCDFIO cimport NetCDFIO_Stats
from scipy.integrate import odeint
from thermodynamic_functions import entropy_from_tp, eos, alpha_from_tp
include 'parameters.pxi'


# cdef extern from "thermodynamic_functions.h":
#     inline double qt_from_pv(double p0, double pv)
'''
Idea:
1) read in case specific surface values
2) compute remaining surface values, using thermodynamic functions (e.g. compute surface entropy from temperature and moisture)
    Note: use same thermodynamic functions for dry and moist conditions, since equivalent with qt=ql=qi=0
3) compute pressure profile by integrating the hydrostatic equation
4) compute other thermodynamic profiles
'''

cdef class ReferenceState:
    def __init__(self, Grid.Grid Gr ):

        self.p0 = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.p0_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.alpha0 = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.alpha0_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.rho0 = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.rho0_half = np.zeros(Gr.nzg, dtype=np.double, order='c')

        return

    # def initialize(self, Grid.Grid Gr, Thermodynamics, NetCDFIO_Stats NS):
    def initialize(self, Grid.Grid Gr, NetCDFIO_Stats NS):
        '''
        Initilize the reference profiles. The function is typically called from the case specific initialization
        fucntion defined in Initialization.pyx
        :param Gr: Grid class
        :param Thermodynamics: Thermodynamics class
        :param NS: StatsIO class
        :return:
        '''
        print('Reference state initialization')

        # ((1)) t --> s
        qlg = 0.0
        qig = 0.0
        self.sg = entropy_from_tp(self.Pg, self.Tg, self.qtg, qlg, qig)

        # ((2)) Pressure Profile (Hydrostatic)
        # (2a) Construct arrays for integration points
        z = np.array(Gr.z[Gr.gw - 1:-Gr.gw + 1])
        z_half = np.append([0.0], np.array(Gr.z_half[Gr.gw:-Gr.gw]))

        # (2b) Perform the integration of the hydrostatic equation to determine the reference pressure
        #       dp/dz = - rho*g --> dp/p = -g/(R*T) = RHS       (becomes more complicated for moist thermodynamics)
        #       --> log(p/p0) = odeint(RHS, log(p0), z)
        #       (i) Form the right hand side of the hydrostatic equation to calculate log(p)
        def rhs(p,z):
            # return of structure a necessary for multiple return from cython function
            a = eos(np.exp(p), self.qtg, self.sg)       # # T, ql, qi = Thermodynamics.eos(np.exp(p), self.sg, self.qtg)
            qi = 0.0
            ql = a['ql']
            T = a['T']
            return -g / (Rd * T * (1.0 - self.qtg + eps_vi * (self.qtg - ql - qi)))

        #       (ii) We are integrating the log pressure so need to take the log of the surface pressure
        p0 = np.log(self.Pg)
        p = np.zeros(Gr.nzg, dtype=np.double, order='c')
        p_half = np.zeros(Gr.nzg, dtype=np.double, order='c')

        #       (iii) Integrate for log(p)
        p[Gr.gw - 1:-Gr.gw +1] = odeint(rhs, p0, z, hmax=1.0)[:, 0]     # only unsaturated eos in DCBLSoares
        p_half[Gr.gw:-Gr.gw] = odeint(rhs, p0, z_half, hmax=1.0)[1:, 0]     # only unsaturated eos in DCBLSoares

        # (2c) Set boundary conditions
        p[:Gr.gw - 1] = p[2 * Gr.gw - 2:Gr.gw - 1:-1]
        p[-Gr.gw + 1:] = p[-Gr.gw - 1:-2 * Gr.gw:-1]

        p_half[:Gr.gw] = p_half[2 * Gr.gw - 1:Gr.gw - 1:-1]
        p_half[-Gr.gw:] = p_half[-Gr.gw - 1:-2 * Gr.gw - 1:-1]

        # (2d) compute p from log(p)
        p = np.exp(p)
        p_half = np.exp(p_half)

        # ((3)) Compute reference state thermodynamic profiles
        cdef double[:] p_ = p
        # cdef double[:] p_ = np.exp(p)
        cdef double[:] p_half_ = p_half
        # cdef double[:] p_half_ = np.exp(p_half)
        cdef double[:] temperature = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] temperature_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] alpha = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] alpha_half = np.zeros(Gr.nzg, dtype=np.double, order='c')

        cdef double[:] ql = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qi = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qv = np.zeros(Gr.nzg, dtype=np.double, order='c')

        cdef double[:] ql_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qi_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qv_half = np.zeros(Gr.nzg, dtype=np.double, order='c')


        for k in xrange(Gr.nzg):
            a = eos(p_[k], self.qtg, self.sg)       # temperature[k], ql[k], qi[k] = Thermodynamics.eos(p_[k], self.sg, self.qtg)
            temperature[k] = a['T']
            ql[k] = a['ql']
            qi[k] = 0.0
            qv[k] = self.qtg - (ql[k] + qi[k])
            alpha[k] = alpha_from_tp(p_[k], temperature[k], self.qtg, qv[k])        # alpha[k] = Thermodynamics.alpha(p_[k], temperature[k], self.qtg, qv[k])

            a = eos(p_half_[k], self.qtg, self.sg)      # temperature_half[k], ql_half[k], qi_half[k] = Thermodynamics.eos(p_half_[k], self.sg, self.qtg)
            temperature_half[k] = a['T']
            ql_half[k] = a['ql']
            qi_half[k] = 0.0
            qv_half[k] = self.qtg - (ql_half[k] + qi_half[k])
            alpha_half[k] = alpha_from_tp(p_half_[k], temperature_half[k], self.qtg, qv_half[k])        # alpha_half[k] = Thermodynamics.alpha(p_half_[k], temperature_half[k], self.qtg, qv_half[k])


        # ((4)) Sanity check to make sure that the Reference State entropy profile is uniform following saturation adjustment
        cdef double s
        for k in xrange(Gr.nzg):
            s = entropy_from_tp(p_half[k],temperature_half[k],self.qtg,ql_half[k],qi_half[k])       # s = Thermodynamics.entropy(p_half[k],temperature_half[k],self.qtg,ql_half[k],qi_half[k])
            if np.abs(s - self.sg)/self.sg > 0.01:
                print('Error in reference profiles entropy not constant !')
                print('Likely error in saturation adjustment')
                print('Kill Simulation Now!')
                sys.exit()

        self.alpha0_half = alpha_half
        self.alpha0 = alpha
        self.p0 = p_
        self.p0_half = p_half
        self.rho0 = 1.0 / np.array(self.alpha0)
        self.rho0_half = 1.0 / np.array(self.alpha0_half)

        # Write reference profiles to StatsIO
        NS.add_reference_profile('alpha0', Gr)
        NS.write_reference_profile('alpha0', alpha_half[Gr.gw:-Gr.gw])
        NS.add_reference_profile('p0', Gr)
        NS.write_reference_profile('p0', p_half[Gr.gw:-Gr.gw])
        NS.add_reference_profile('rho0', Gr)
        NS.write_reference_profile('rho0', 1.0 / np.array(alpha_half[Gr.gw:-Gr.gw]))
        NS.add_reference_profile('temperature0', Gr)
        NS.write_reference_profile('temperature0', temperature_half[Gr.gw:-Gr.gw])
        NS.add_reference_profile('ql0', Gr)
        NS.write_reference_profile('ql0', ql_half[Gr.gw:-Gr.gw])
        NS.add_reference_profile('qv0', Gr)
        NS.write_reference_profile('qv0', qv_half[Gr.gw:-Gr.gw])
        NS.add_reference_profile('qi0', Gr)
        NS.write_reference_profile('qi0', qi_half[Gr.gw:-Gr.gw])

        return













    # cpdef restart(self, Grid.Grid Gr, Restart.Restart Re):
    #     Re.restart_data['Ref'] = {}
    #
    #     Re.restart_data['Ref']['p0'] = np.array(self.p0)
    #     Re.restart_data['Ref']['p0_half'] = np.array(self.p0_half)
    #     Re.restart_data['Ref']['alpha0'] = np.array(self.alpha0)
    #     Re.restart_data['Ref']['alpha0_half'] = np.array(self.alpha0_half)
    #
    #     Re.restart_data['Ref']['p0_global'] = np.array(self.p0_global)
    #     Re.restart_data['Ref']['p0_half_global'] = np.array(self.p0_half_global)
    #     Re.restart_data['Ref']['alpha0_global'] = np.array(self.alpha0_global)
    #     Re.restart_data['Ref']['alpha0_half_global'] = np.array(self.alpha0_half_global)
    #
    #     Re.restart_data['Ref']['Tg'] = self.Tg
    #     Re.restart_data['Ref']['Pg'] = self.Pg
    #     Re.restart_data['Ref']['sg'] = self.sg
    #     Re.restart_data['Ref']['qtg'] = self.qtg
    #     Re.restart_data['Ref']['u0'] = self.u0
    #     Re.restart_data['Ref']['v0'] = self.v0
    #
    #     return


    # cpdef init_from_restart(self, Grid.Grid Gr, Restart.Restart Re):
    #
    #     self.Tg = Re.restart_data['Ref']['Tg']
    #     self.Pg = Re.restart_data['Ref']['Pg']
    #     self.sg = Re.restart_data['Ref']['sg']
    #     self.qtg = Re.restart_data['Ref']['qtg']
    #     self.u0 = Re.restart_data['Ref']['u0']
    #     self.v0 = Re.restart_data['Ref']['v0']
    #
    #     self.p0 = Re.restart_data['Ref']['p0']
    #     self.p0_half = Re.restart_data['Ref']['p0_half']
    #     self.alpha0 = Re.restart_data['Ref']['alpha0']
    #     self.alpha0_half = Re.restart_data['Ref']['alpha0_half']
    #     self.rho0 = 1.0 / Re.restart_data['Ref']['alpha0']
    #     self.rho0_half = 1.0 / Re.restart_data['Ref']['alpha0_half']
    #
    #     self.p0_global = Re.restart_data['Ref']['p0_global']
    #     self.p0_half_global = Re.restart_data['Ref']['p0_half_global']
    #     self.alpha0_global = Re.restart_data['Ref']['alpha0_global']
    #     self.alpha0_half_global = Re.restart_data['Ref']['alpha0_half_global']
    #     self.rho0_global = 1.0 / Re.restart_data['Ref']['alpha0_global']
    #     self.rho0_half_global = 1.0 / Re.restart_data['Ref']['alpha0_half_global']

    #     return



