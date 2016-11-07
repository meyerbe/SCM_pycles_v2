#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from Grid cimport Grid
cimport PrognosticVariables
from PrognosticVariables cimport MeanVariables
from PrognosticVariables cimport SecondOrderMomenta
from ReferenceState cimport ReferenceState
from TimeStepping cimport TimeStepping
# cimport DiagnosticVariables
# cimport Surface
# from NetCDFIO cimport NetCDFIO_Stats
from libc.math cimport exp, sqrt
cimport numpy as np
import numpy as np
import pylab as plt
import cython
include 'parameters.pxi'
from thermodynamic_functions import latent_heat, exner


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
    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1):
        return
    # cpdef update(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
    #     print('Turb None: update')
    #     return
    # cpdef stats_io(self):
    #     return


cdef class TurbulenceBase:
    def __init__(self,namelist):
        print('initializing Turbulence Base')
        return

    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1):
        self.tendencies_M1 = np.zeros((M1.nv,Gr.nzg),dtype=np.double,order='c')
        print('Initialize Turbulence Base')
        return

    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turbulence Base: update')
        return

    cpdef update_M1(self,Grid Gr, ReferenceState Ref, TimeStepping TS, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turb: update M1')
        '''
        uw      --> up
        vw      --> vp
        ww      --> wp, wth
        wth      --> thp, thth
        wqt      --> qtp
        pu, pv, pw, pth, pqt (pressure correlations)
        uu,vv  (for TKE)
        '''
        cdef:
            Py_ssize_t k, m, n, var_index
            double [:] th0 = Ref.th0       # !!! right interfaces ???
            double dzi = Gr.dzi
            double [:] alpha0 = Ref.alpha0
            double [:] rho0_half = Ref.rho0_half
            # double [:] rho0 = Ref.rho0
            double [:,:] tendencies_M1 = self.tendencies_M1
            # double [:] temp = np.zeros(shape=alpha0.shape[0])

        # problem: rho0_half[gw+nz] = rho0_half[gw+nz-1] --> gives tendencies[:,gw+nz-1] = 0
        # --> changed in Reference State
        # for k in xrange(Gr.gw):
        #     rho0_half[Gr.gw+Gr.nz+k] = 0.5*(rho0[Gr.gw+Gr.nz+k] + rho0[Gr.gw+Gr.nz+k-1])
        # rho0_half[Gr.gw+Gr.nz] = 0.5*(rho0[Gr.gw+Gr.nz] + rho0[Gr.gw+Gr.nz-1])

        # temp = tendencies_M1[2,:]
        # (1) vertical edd fluxes: in all prognostic variables
        for var in M1.name_index.keys():
            var_index = M1.name_index[var]
            if var=='u' or var=='v':
                m = M1.name_index[var]
                n = M1.name_index['w']
            else:
                m = M1.name_index['w']
                n = M1.name_index[var]
            # print('Turb.udpate_M1: ', var, var_index, m, n)
            for k in xrange(1,Gr.nzg-1):
                # temp[k] = rho0_half[k+1]*M2.values[m,n,k+1]-rho0_half[k]*M2.values[m,n,k]
                # M1.tendencies[m,k] -= alpha0[k]*dzi*(rho0_half[k+1]*M2.values[m,n,k+1]-rho0_half[k]*M2.values[m,n,k])
                tendencies_M1[var_index,k] -= alpha0[k]*dzi*(rho0_half[k+1]*M2.values[m,n,k+1]-rho0_half[k]*M2.values[m,n,k])       # correct

        # self.plot_var('vertflux', tendencies_M1, Gr, Ref, TS, M1, M2)
        with nogil:
            for n in xrange(M1.nv):
                for k in xrange(Gr.nzg):
                    M1.tendencies[n,k] += tendencies_M1[n,k]
                    tendencies_M1[n,k] = 0.0


        # (2) Buoyancy Flux: in w
        # --> mean buoyancy approximated as zero (accurate if <rho> agrees with the reference state)


        # (3) Pressure ???
        # ???

        return


    cpdef stats_io(self):
        return





cdef class Turbulence2ndOrder(TurbulenceBase):
    # (1) Advection: MA.update_M2, SA.update_M2
    # (2) Diffusion: SD.update_M2, MD.update_M2
    # (3) Pressure: ?
    # (4) Third and higher order terms (first guess set to zero)
    def __init__(self,namelist):
        self.buoyancy = None
        return


    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1):
        print('Initializing Turbulence 2nd')
        self.buoyancy = np.zeros((M1.nv,Gr.nzg),dtype=np.double,order='c')
        self.tendencies_M1 = np.zeros((M1.nv,Gr.nzg),dtype=np.double,order='c')
        self.tendencies_M2 = np.zeros((M1.nv+1,M1.nv+1,Gr.nzg),dtype=np.double,order='c')
        return


    cpdef update_M2(self, Grid Gr, ReferenceState Ref, TimeStepping TS, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turbulence 2nd order: update')
        cdef double [:,:,:] tend = self.tendencies_M2
        for m in xrange(M2.nv):
            for n in xrange(M2.nv):
                for k in xrange(Gr.nzg):
                    tend[m,n,k] = 0.0
        print('max M2.tend', np.amax(self.tendencies_M2))

        temp1 = M2.tendencies

        # (1) Advection
        # (2) Pressure
        # (3) Buoyancy
        # (4) Diffusion
        # (5) Third and higher order terms (first guess set to zero)
        M2.plot_tendencies('01_init',Gr,TS)
        self.advect_M2_local(Gr, Ref, TS, M1, M2)
        M2.plot_tendencies('02_advect',Gr,TS)
        temp2 = M2.tendencies
        # self.plot_all('uw', Gr, TS, M1, M2, temp[0,2,:], 0, 2)
        # self.plot_all('vw', Gr, TS, M1, M2, temp[1,2,:], 1, 2)
        # self.plot_all('ww', Gr, TS, M1, M2, temp[2,2,:], 2, 2)
        # self.plot_all('wth', Gr, TS, M1, M2, temp[2,3,:], 2, 3)
        self.pressure_correlations_Andre(Gr,Ref,TS,M1, M2)
        # temp3 = M2.tendencies
        # temp4 = temp3-temp2
        M2.plot_tendencies('03_pressureAndre',Gr,TS)
        # self.plot_var2('difference',0,0,M2.tendencies[0,0,:],temp4[0,0,:],Gr,Ref,TS,M1,M2)
        # self.plot_var2('difference',0,2,M2.tendencies[0,2,:],temp4[0,2,:],Gr,Ref,TS,M1,M2)
        # self.plot_var2('difference',2,2,M2.tendencies[2,2,:],temp4[2,2,:],Gr,Ref,TS,M1,M2)
        # self.plot_var2('difference',2,3,M2.tendencies[2,3,:],temp4[2,3,:],Gr,Ref,TS,M1,M2)
        # self.plot_var2('difference',3,3,M2.tendencies[3,3,:],temp4[3,3,:],Gr,Ref,TS,M1,M2)

        # self.plot_all('uw', Gr, TS, M1, M2, temp[0,2,:], 0, 2)
        # self.plot_all('vw', Gr, TS, M1, M2, temp[1,2,:], 1, 2)
        # self.plot_all('ww', Gr, TS, M1, M2, temp[2,2,:], 2, 2)
        # self.plot_all('wth', Gr, TS, M1, M2, temp[2,3,:], 2, 3)
        ## self.pressure_correlations_Cheng(Gr, M1, M2)
        ## self.pressure_correlations_Mironov(Gr, M1, M2)
        # self.buoyancy_update(Gr, Ref, TS, M1, M2)
        # M2.plot_tendencies('04_buoyancy',Gr,TS)
        return


    cpdef advect_M2_local(self, Grid Gr, ReferenceState Ref, TimeStepping TS, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turb update: advect M2 local!')
        # implemented for staggered grid

        cdef:
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            Py_ssize_t w_index = M1.name_index['w']
            Py_ssize_t th_index = M1.name_index['th']
            Py_ssize_t qt_index

            double [:] u = M1.values[u_index,:]
            double [:] v = M1.values[v_index,:]
            double [:] w = M1.values[w_index,:]
            double [:] th = M1.values[th_index,:]

            # double [:] flux = np.zeros(Gr.nzg,dtype=np.double,order='c')
            double [:,:,:] flux = np.zeros((M2.nv,M2.nv,Gr.nzg),dtype=np.double,order='c')
            double [:,:,:] aux = np.zeros((M2.nv,M2.nv,Gr.nzg),dtype=np.double,order='c')
            double [:,:,:] tendencies = self.tendencies_M2
            double [:,:,:] values = M2.values
            double [:] rho0 = Ref.rho0
            double [:] rho0_half = Ref.rho0_half
            double [:] alpha0_half = Ref.alpha0_half

            double dzi = Gr.dzi
            Py_ssize_t gw = Gr.gw
            Py_ssize_t m, n, k, var_shift


        # (i) advection by mean vertical velocity
        #       (interpolate only M2 in fluxes (w on staggered grid))
        for m in xrange(M2.nv):
            for n in xrange(m,M2.nv):
                for k in xrange(Gr.nzg-1):
                    # flux[k] = rho0[k]*w[k]*0.5*(M2.values[m,n,k+1]+M2.values[m,n,k])
                    flux[m,n,k] = rho0[k]*w[k]*0.5*(M2.values[m,n,k+1]+M2.values[m,n,k])
                for k in xrange(1,Gr.nzg-1):
                    # tendencies[m,n,k] = rho0_half[k]*0.5*(w[k+1]+w[k])*dzi2*(rho0_half[k+1]*M2.values[m,n,k+1]-rho0_half[k-1]*M2.values[m,n,k-1])
                    # tendencies[m,n,k] += - alpha0_half[k]*dzi*(flux[k]-flux[k-1])
                    tendencies[m,n,k] += - alpha0_half[k]*dzi*(flux[m,n,k]-flux[m,n,k-1])



        print('M2 advection tendencies:', np.amax(np.abs(tendencies)), np.amax(np.abs(self.tendencies_M2)) )
        self.plot_var2('advection_uu', 0,0,flux[0,0,:], tendencies[0,0,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('advection_uw', 0,2,flux[0,2,:], tendencies[0,2,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('advection_ww', 2,2,flux[2,2,:], tendencies[2,2,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('advection_wth', 2,3,flux[2,3,:], tendencies[2,3,:], Gr, Ref, TS, M1, M2)
        # self.plot_var('advection', tendencies[2,:,:], Gr, Ref, TS, M1, M2)
        for m in xrange(M2.nv):
            for n in xrange(m,M2.nv):
                for k in xrange(Gr.nzg):
                    M2.tendencies[m,n,k] += tendencies[m,n,k]
                    tendencies[m,n,k] = 0.0
        print('Turb.update_M2 after advection by M1', np.amax(tendencies))

        M2.plot_tendencies('02a_advect',Gr,TS)
        M2.plot_nogw_tendencies('02a_advect',Gr,TS)

        # (ii) advection by M2
        #     u,v,w,th,qt: on w-grid
        #     M2: on shifted grid
        for k in xrange(1,Gr.nzg):
            tendencies[u_index, u_index, k] += -2*values[u_index,w_index,k] * (u[k]-u[k-1])*dzi
            tendencies[v_index, v_index, k] += -2*values[v_index,w_index,k] * (v[k]-v[k-1])*dzi
            tendencies[u_index, w_index, k] += -values[w_index,w_index,k]*(u[k]-u[k-1])*dzi - values[u_index,w_index,k]*(w[k]-w[k-1])*dzi
            tendencies[v_index, w_index, k] += -values[w_index,w_index,k]*(v[k]-v[k-1])*dzi - values[v_index,w_index,k]*(w[k]-w[k-1])*dzi
            tendencies[w_index, w_index, k] += -2*values[w_index,w_index,k]*(w[k]-w[k-1])*dzi
            tendencies[u_index, th_index, k] += -values[w_index,th_index,k]*(u[k]-u[k-1])*dzi - values[u_index,w_index,k]*(th[k]-th[k-1])*dzi
            tendencies[v_index, th_index, k] += -values[w_index,th_index,k]*(v[k]-v[k-1])*dzi - values[v_index,w_index,k]*(th[k]-th[k-1])*dzi
            tendencies[w_index, th_index, k] += -values[w_index,th_index,k]*(w[k]-w[k-1])*dzi - values[w_index,w_index,k]*(th[k]-th[k-1])*dzi
            tendencies[th_index, th_index, k] += -2*values[w_index,th_index,k]*(th[k]-th[k-1])*dzi

        if 'qt' in M1.index_name:
            qt_index = M1.name_index['qt']
            qt = M1.values[qt_index,:]
            for k in xrange(1,Gr.nzg-1):
                tendencies[u_index, qt_index, k] += -values[w_index,qt_index,k]*(u[k]-u[k-1])*dzi - values[u_index,w_index,k]*(qt[k]-qt[k-1])*dzi
                tendencies[v_index, qt_index, k] += -values[w_index,qt_index,k]*(v[k]-v[k-1])*dzi - values[v_index,w_index,k]*(qt[k]-qt[k-1])*dzi
                tendencies[w_index, qt_index, k] += -values[w_index,qt_index,k]*(w[k]-w[k-1])*dzi - values[w_index,w_index,k]*(qt[k]-qt[k-1])*dzi
                tendencies[qt_index, qt_index, k] += -2*values[w_index,qt_index,k]*(qt[k]-qt[k-1])*dzi

        for m in xrange(M2.nv):
            for n in xrange(m,M2.nv):
                for k in xrange(Gr.nzg):
                    M2.tendencies[m,n,k] += tendencies[m,n,k]
                    tendencies[m,n,k] = 0.0
        # # print('---- M2 advection tendencies:', np.amax(np.abs(tendencies)), np.amax(np.abs(self.tendencies_M2)) )
        # # self.plot_var('advection2', tendencies[2,:,:], Gr, Ref, TS, M1, M2)

        self.plot_var2('advection_ww2', 2,2,tendencies[2,2,:], tendencies[2,2,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('advection_uw2', 0,2,tendencies[0,2,:], tendencies[0,2,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('advection_wth2', 2,3,tendencies[2,3,:], tendencies[2,3,:], Gr, Ref, TS, M1, M2)

        M2.plot_tendencies('02b_advect',Gr,TS)

        return






    cpdef pressure_correlations_Andre(self, Grid Gr, ReferenceState Ref, TimeStepping TS, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        # '''following André (1978), based on Launder (1975) and Rotta (1951)'''
        # !!! only dry thermodynamics !!!
        # !!! only expression for vertical moisture flux (no horizontal fluxes) !!!
        print('Turb update: pressure correlations André')

        # time = 0

        cdef:
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            Py_ssize_t w_index = M1.name_index['w']
            Py_ssize_t th_index = M1.name_index['th']
            Py_ssize_t qt_index
            double [:] u = M1.values[u_index,:]
            double [:] v = M1.values[v_index,:]
            double [:] w = M1.values[w_index,:]
            double [:,:,:] M2_values = M2.values
            double [:,:,:] tendencies = self.tendencies_M2

            double [:,:,:] P = np.zeros(shape=(M1.nv,M1.nv,Gr.nzg),dtype=np.double,order='c')
            double [:] P_tke = np.zeros((Gr.nzg),dtype=np.double,order='c')
            double [:] tke = np.zeros((Gr.nzg),dtype=np.double,order='c')

            double lb=0.0
            double ld=0.0
            double l0, karman, c1, delta
            double [:] l = np.zeros((Gr.nzg),dtype=np.double,order='c')
            double [:] epsilon = np.zeros((Gr.nzg),dtype=np.double,order='c')

            Py_ssize_t k,m,n
            Py_ssize_t n_vel = M1.nv_velocities
            double dzi = Gr.dzi
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t gw = Gr.gw

            double alpha = 1/285    # thermal expansion coefficient
            double c4 = 4.5
            double c5 = 0.0
            double c6 = 4.85
            double c7 = 1.0-0.125*c6


        print('Turb.update_M2 pressure', np.amax(tendencies))

        print('eps 1: ', np.amax(epsilon), np.amin(epsilon))
        print('tke 1: ', np.amax(tke), np.amin(tke), np.amax(tke[gw:nzg]), np.amin(tke[gw:nzg]))

        # (1) TKE and TKE generation rate
        # Note: for updating M1, the first upper ghost point of M2.values is required
        #       --> need to compute the generation rates on k\in[0,nz]
        for k in xrange(1,nzg):
            # (a) Kinetic Energy ---> Diagnostic Variable (needed elsewhere?!?!)
            tke[k] = np.sqrt(M2_values[0,0,k] + M2_values[1,1,k] + M2_values[2,2,k])
            # (b) TKE generation rate
            P_tke[k] = alpha*g*M2_values[w_index,th_index,k] \
                        - M2_values[u_index,w_index,k]*(u[k]-u[k-1])*dzi \
                        - M2_values[v_index,w_index,k]*(v[k]-v[k-1])*dzi

        print('tke 2: ', np.amax(tke), np.amin(tke), np.amax(tke[gw:nzg]), np.amin(tke[gw:nzg]))
        if np.isnan(tke).any():
            print('tke 2: tke is nan')
        # if np.isnan(P_tke).any():
        #     print('PPPPP: P_tke is nan')


        # (2) Energy Dissipation
        # lb: Mixing length for neutral and unstable conditions (Blackadar, 1962)
        # ld: characteristic lengthfor stable stratification
        karman = 0.35           # von Karman constant
        l0 = 15                 # mixing length far above the ground, take 15m (Jieliu, 2011)
        for k in xrange(gw,nzg-1):
            lb = karman*k/(1+karman*k/l0)       # stable stratification; k>0 required
            # print(k, 'lb', lb)
            delta = (M1.values[th_index,k+1]-M1.values[th_index,k])*dzi
            if delta > 0.0:
                ld = 0.75*(np.sqrt(tke[k]))*1/np.sqrt(alpha*g*(M1.values[th_index,k+1]-M1.values[th_index,k])*dzi)      # neutral
                l[k] = np.minimum(lb,ld)
            else:
                print(k, 'delta is zero', delta)
                l[k] = lb
            c1 = 0.019+0.051*l[k]/lb
            epsilon[k] = c1*np.sqrt(tke[k]*tke[k]*tke[k])
            # if epsilon[k] == 0.0 or np.isnan(epsilon[k]):
            #     print('epsilon is zero or nan', epsilon[k], k)
            #     print('c1, lb, ld, l[k], tke[k]: ', c1, lb, ld, l[k], tke[k], k)
            if epsilon[k] == - float('Inf'):
                print('espilon is -infinity: k, eps', k, Gr.nz, gw, epsilon[k], c1, lb, ld, l[k], tke[k])

        # if np.isnan(epsilon).any():
        #     print('PPPPP: epsilon is nan')
        print('eps 2:', np.amax(epsilon), np.amin(epsilon), np.isnan(epsilon).any())
        print('tke 3:', np.amax(tke), np.amin(tke))

        # (3) Momentum Fluxes: generation rate and tendency
        # (3a) diagonal elements
        for m in xrange(n_vel):
            for k in xrange(gw,nzg):
                P[m,w_index,k] = alpha*g*(M2_values[m,th_index,k])
                P[w_index,m,k] = alpha*g*(M2_values[m,th_index,k])
            for k in xrange(nzg):
                # M2.tendencies[n,n,k] += 2/3*( c4*epsilon[k] + c5*P_tke[k])
                tendencies[m,m,k] += 2./3.*( c4*epsilon[k] + c5*P_tke[k] )
        # (3b) off-diagnoal elements
            for n in xrange(m,n_vel):
                for k in xrange(gw,nzg):
                    P[m,n,k] -= M2_values[m,w_index,k]*(M1.values[n,k]-M1.values[n,k-1])*dzi \
                                + M2_values[n,w_index,k]*(M1.values[m,k]-M1.values[m,k-1])*dzi
                    # if np.isnan(P[m,n,k]):
                    #     print('....... P is nan, ', m,n,k)
                for k in xrange(gw,nzg):
                    if tke[k] > 0:
                        # M2.tendencies[m,n,k] += -c4*epsilon[k]/tke[k]*M2.values[m,n,k] - c5*P[m,n,k]
                        tendencies[m,n,k] += -c4*epsilon[k]/tke[k]*M2_values[m,n,k] - c5*P[m,n,k]
                    else:
                        # M2.tendencies[m,n,k] -= c5*P[m,n,k]     # c5=0
                        tendencies[m,n,k] -= c5*P[m,n,k]     # c5=0
        # if np.isnan(P).any():
        #     print('PPPPP: P is nan')
        # if np.isnan(M2.tendencies).any():
        #     print('PPPPP: M2.tend is nan')

        # (4) Heat Flux generation rate and tendency
        with nogil:
            for n in xrange(n_vel):
                for k in xrange(gw,nzg):
                    P[n,th_index,k] = alpha*g*M2_values[w_index,th_index,k] \
                                      - M2_values[u_index,w_index,k]*(u[k]-u[k-1])*dzi \
                                      - M2_values[v_index,w_index,k]*(v[k]-v[k-1])*dzi
                for k in xrange(gw,nzg):
                    if tke[k] > 0:
                        # M2.tendencies[n,th_index,k] += -c6*epsilon[k]/tke[k]*M2.values[n,th_index,k] - c7*P[n,th_index,k]
                        tendencies[n,th_index,k] += -c6*epsilon[k]/tke[k]*M2.values[n,th_index,k] - c7*P[n,th_index,k]
                    else:
                        # M2.tendencies[i,th_index,k] -= c7*P[i,th_index,k]
                        tendencies[n,th_index,k] -= c7*P[n,th_index,k]
        if np.isnan(P).any():
            print('PPPPP: P is nan after')
        if np.isnan(P_tke).any():
            print('PPPPP: P_tke is nan')
        if np.isnan(tke).any():
            print('PPPPP: tke is nan')
        if np.isnan(tendencies).any():
            print('PPPPP: tendencies is nan')

        # (5) Moisture Flux generation rate and tendency
        if 'qt' in M1.name_index:
            qt_index = M1.name_index['qt']
            for k in xrange(Gr.nzg-1):
                if th_index < qt_index:
                    P[w_index,qt_index,k] = alpha*g*M2.values[th_index,qt_index,k]
                else:
                    P[w_index,qt_index,k] = alpha*g*M2.values[qt_index,th_index,k]
            for k in xrange(Gr.nzg-1):
                # M2.tendencies[w_index,qt_index,k] = -c6*epsilon[k]/tke[k]*M2.values[w_index,qt_index,k] \
                #                                     - c7*P[w_index,qt_index,k]
                tendencies[w_index,qt_index,k] = -c6*epsilon[k]/tke[k]*M2.values[w_index,qt_index,k] \
                                                    - c7*P[w_index,qt_index,k]
        for m in xrange(n_vel):
            for n in xrange(n_vel):
                for k in xrange(1,nzg):
                    M2.tendencies[m,n,k] += tendencies[m,n,k]

        self.plot_var2('pressure_uu', 0,0, tendencies[0,0,:], M2.tendencies[0,0,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('pressure_vv', 1,1, tendencies[1,1,:], M2.tendencies[1,1,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('pressure_ww', 2,2, tendencies[2,2,:], M2.tendencies[2,2,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('pressure_uw', 0,2, tendencies[0,2,:], M2.tendencies[0,2,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('pressure_wu', 2,0, tendencies[2,0,:], M2.tendencies[2,0,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('pressure_wth', 2,3, tendencies[2,3,:], M2.tendencies[2,3,:], Gr, Ref, TS, M1, M2)

        self.plot_var2('pressure_Puu', 0,0, P[0,0,:], tendencies[0,0,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('pressure_Pww', 2,2, P[2,2,:], tendencies[2,2,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('pressure_Puw', 0,2, P[0,2,:], tendencies[0,2,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('pressure_Pwth', 2,3, P[2,3,:], tendencies[2,3,:], Gr, Ref, TS, M1, M2)
        self.plot_var2('pressure_Ptke', 0,0, P_tke[:], tke[:], Gr, Ref, TS, M1, M2)
        self.plot_var2('pressure_eps', 0,2, epsilon[:], tendencies[0,2,:], Gr, Ref, TS, M1, M2)

        # time += 1
        return


    cpdef pressure_correlations_Mironov(self, Grid Gr, ReferenceState Ref, TimeStepping TS, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        '''following Mironov (2009), based on André (1978), Launder (1975) and Rotta (1951)'''
        cdef:
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            Py_ssize_t w_index = M1.name_index['w']
            # Py_ssize_t th_index = M1.name_index['th']

            Py_ssize_t k, shift
            double dzi2 = 0.5*Gr.dzi

            double [:] tke = np.zeros((Gr.nzg),dtype=np.double,order='c')
        #     # double [:] sxz = np.zeros((Gr.nzg),dtype=np.double,order='c')
        #     # double [:] syz = np.zeros((Gr.nzg),dtype=np.double,order='c')
        #     # double [:] szz = np.zeros((Gr.nzg),dtype=np.double,order='c')
        #     # double [:] wxz = np.zeros((Gr.nzg),dtype=np.double,order='c')
        #     # double [:] wyz = np.zeros((Gr.nzg),dtype=np.double,order='c')
        #     # double [:] wzz = np.zeros((Gr.nzg),dtype=np.double,order='c')
        #
        #     double Sxz, Syz, Szz, Wxz, Wzx, Wyz, Wzy, Wzz
        #
        #     double Ctu = 0.0
        #     double Cs1u = 0.0
        #     double Cs2u = 0.0
        #     double Cbu = 0.0
        #     double Ccu = 0.0
        #     double Cttheta = 0.0
        #     double Cbtheta = 0.0
        #
        # # M2.name_index:    {'pv': 6, 'pw': 7, 'pu': 5, 'uu': 0, 'vth': 9, 'uth': 8, 'ww': 4, 'wv': 3, 'wu': 2, 'vv': 1, 'wth': 10}
        # # M2.name_index.keys():     gives only names (=keys)
        # # M2.name_index['uu']:      gives the index of variable with name (or key) 'uu'
        # # M2.index_name[0]:         gives name (or key) at position 0

        for k in xrange(Gr.nzg):
            # Kinetic Energy ---> Diagnostic Variable (needed elsewhere?!?!)
            tke[k] = np.sqrt(M2.values[0,0,k] + M2.values[1,1,k] + M2.values[2,2,k])
        # # for k in xrange(1,Gr.nzg-1):
        # #     # Mean Shear ---> Diagnostic Variable (needed elsewhere?!?!)
        # #     sxz[k] = 0.5*( (M1.values[u_shift+k+1]-M1.values[u_shift+k-1])*dzi2 )
        # #     syz[k] = 0.5*( (M1.values[v_shift+k+1]-M1.values[v_shift+k-1])*dzi2 )
        # #     szz[k] = (M1.values[w_shift+k+1]-M1.values[w_shift+k-1])*dzi2
        # #     wxz[k] = 0.5*( (M1.values[u_shift+k+1]-M1.values[u_shift+k-1])*dzi2 )
        # #     wyz[k] = 0.5*( (M1.values[v_shift+k+1]-M1.values[v_shift+k-1])*dzi2 )
        # #     wzz[k] = (M1.values[w_shift+k+1]-M1.values[w_shift+k-1])*dzi2
        #
        # print(M2.name_index.keys())
        # print(M2.name_index)
        # print(M2.name_index['uu'])
        # print(M2.index_name[0])
        # # for var in M2.name_index.keys():        #go through all var-names
        # #     index = M2.name_index[var]          # get index of var
        # #     shift = M2.get_varshift(Gr, var)
        # #     for k in xrange(1,Gr.nzg-1):
        # #         # departure-from-isotropy tensor
        # #         if var in ['uu', 'vv', 'ww']:
        # #             a = 2*M2.values[shift+k]/tke[k] - 2.0/3.0
        # #         else:
        # #             a = 2*M2.values[shift+k]/tke[k]
        # #
        # #         # Mean Shear ---> Diagnostic Variable (needed elsewhere?!?!)
        # #         Sxz = 0.5*( (M1.values[u_shift+k+1]-M1.values[u_shift+k-1])*dzi2 )
        # #         Syz = 0.5*( (M1.values[v_shift+k+1]-M1.values[v_shift+k-1])*dzi2 )
        # #         Szz = (M1.values[w_shift+k+1]-M1.values[w_shift+k-1])*dzi2
        # #         Wzx = 0.5*( (M1.values[u_shift+k+1]-M1.values[u_shift+k-1])*dzi2 )
        # #         Wxz = -Wzx
        # #         Wzy = 0.5*( (M1.values[v_shift+k+1]-M1.values[v_shift+k-1])*dzi2 )
        # #         Wyz = -Wzy
        # #         Wzz = (M1.values[w_shift+k+1]-M1.values[w_shift+k-1])*dzi2
        #
        #
        # var = 'uu'
        # index = M2.name_index[var]          # get index of var
        # shift = M2.get_varshift(Gr, var)
        # for k in xrange(1,Gr.nzg-1):
        #     # departure-from-isotropy tensor
        #     a = 2*M2.values[shift+k]/tke[k] - 2.0/3.0
        #     # Mean Shear ---> Diagnostic Variable (needed elsewhere?!?!)
        #     Sxz = 0.5*( (M1.values[u_shift+k+1]-M1.values[u_shift+k-1])*dzi2 )
        #     Wzx = 0.5*( (M1.values[u_shift+k+1]-M1.values[u_shift+k-1])*dzi2 )
        #     Wxz = -Wzx
        #
        #     tend = Ctu
        #
        #
        #
        #     # departure-from-isotropy tensor
        #     # (a) velocities
        #     # i = 0
        #     # i_shift = i*nzg
        #     # a =
        #
        #
        #     M2.tendencies[wu_shift+k] += 0
        #     M2.tendencies[wu_shift+k] += 0
        #     M2.tendencies[wu_shift+k] += 0
        #     M2.tendencies[wu_shift+k] += 0
        #     M2.tendencies[wu_shift+k] += 0
        #     M2.tendencies[wu_shift+k] += 0
        #
        #
        # '''
        # Anelastic 2nd order equation: pressure terms (Rotta '51; Andre '78; Golaz '02a)
        # \partialt \mean{u_i u_j} = \dots + A + B
        # A = \frac{1}{\rhon}\mean{p(\partialj u_i + \partiali u_j)}
        # B = - \sum_{k=1}^3\partialk \mean{(\delta_{jk}u_i + \delta_{ik}u_j)\frac{p}{\rhon} }
        # \partialt \mean{u_i u_j} = \dots + \frac{1}{\rhon}\mean{p(\partialj u_i + \partiali u_j)} - \sum_{k=1}^3\partialk \mean{(\delta_{jk}u_i + \delta_{ik}u_j)\frac{p}{\rhon} }
        #
        # TERM A:
        # constant rho:
        # \frac{1}{\rho}\mean{p(\partialj u_i + \partiali u_j)} - \sum_{k=1}^3\mean{(\delta_{jk}u_i + \delta_{ik}u_j)\frac{p}{\rho} }
        # Anelastic: \rhon(z)
        # \frac{1}{\rhon}\mean{p\left(\partialj u_i + \partiali u_j \right)} = -k_p \frac{\sqrt{E}}{L}\left(\mean{u_iu_j}-\frac23\delta_{ij}e\right)
        #
        # TERM B: ????
        #  \partialk \mean{(\delta_{jk}u_i + \delta_{ik}u_j)\frac{p}{\rhon}} = ???
        # '''

        return

    cpdef pressure_correlations_Cheng(self, Grid Gr, ReferenceState Ref, TimeStepping TS, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        cdef:
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            Py_ssize_t w_index = M1.name_index['w']
        # for k in xrange(Gr.nzg):
        #     Kinetic Energy ---> Diagnostic Variable (needed elsewhere?!?!)
            # tke[k] = np.sqrt(M2.values[0,0,k] + M2.values[1,1,k] + M2.values[2,2,k])
        return


    cpdef stats_io(self):

        return





    '''update buoyancy correlation term of M2'''
    cpdef buoyancy_update(self, Grid Gr, ReferenceState Ref, TimeStepping TS, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        cdef:
            double [:] th0 = Ref.th0       # !!! right interfaces ???
            double [:] p0 = Ref.p0       # !!! right interfaces ???
            double [:] p = p0               # !!! need actual pressure !!!
            double [:] wql = np.zeros((Gr.nzg),dtype=np.double,order='c')       # !!!! how to get <w'ql'>????
            double [:,:] M2_b = np.zeros((M1.nv,Gr.nzg),dtype=np.double,order='c')
            double [:,:,:] tendencies_M2 = self.tendencies_M2

            int nth, nqt, m, k
            int nw = M1.name_index['w']
            double L
            str var

        # Buoyancy Flux: in w
        if 'qt' in M1.name_index.keys():
            nth = M1.name_index['th']
            nqt = M1.name_index['qt']
            # nql = M1.name_index['qt'] --> !!
            # buoyancy[m,k] = <var'th_v'> + (1-ep)/ep*th_0*
            for var in M1.name_index.keys():
                m = M1.name_index[var]
                for k in xrange(Gr.nzg):
                    L = latent_heat(293.0)
                    M2_b[m,k] = M2.values[m,nth,k] + (1-eps_v)/eps_v*th0[k]*M2.values[m,nqt,k] \
                                    + ((L/cpd)*exner(p0[k]/p[k])**(Rd/cpd) - eps_vi*th0[k])*wql[k]
                    # ???? cpd correct in both cases ???
        else:
            nth = M1.name_index['th']
            for var in M1.name_index.keys():
                m = M1.name_index[var]
                for k in xrange(Gr.nzg):
                    M2_b[m,k] = M2.values[m,nth,k]

        self.buoyancy = M2_b

        list = ['uw', 'vw', 'ww', 'wth', 'wqt', 'wp']
        # for n in xrange(M1.nv):
        #     for k in xrange(Gr.nzg):
        #         M1.tendencies[n,k] -= g/th0[k]*self.buoyancy[n,k]

        for n in xrange(M2.nv):
            for k in xrange(Gr.nzg):
                if n < nth:
                    tendencies_M2[n,nw,k] -= g/th0[k]*self.buoyancy[n,k]
                else:
                    tendencies_M2[nw,n,k] -= g/th0[k]*self.buoyancy[n,k]

        self.plot_var('buoyancy', self.buoyancy, Gr, TS, M1, M2)
        # with nogil:
        #     for n in xrange(M2.nv):
        #         for k in xrange(Gr.nzg):
        #             if n < nth:
        #                 M2.tendencies[n,nw,k] += tendencies_M2[n,nw,k]
        #                 tendencies_M2[n,nw,k] = 0.0
        #             else:
        #                 M2.tendencies[nw,n,k] += tendencies_M2[nw,n,k]
        #                 tendencies_M2[nw,n,k] = 0.0


        return







    def plot_var(self, str message, var, Grid Gr, ReferenceState Ref, TimeStepping TS, MeanVariables M1, SecondOrderMomenta M2):
        cdef:
            Py_ssize_t th_varshift = M1.name_index['th']
            Py_ssize_t w_varshift = M1.name_index['w']
            Py_ssize_t v_varshift = M1.name_index['v']
            Py_ssize_t u_varshift = M1.name_index['u']
            double [:] rho0_half = Ref.rho0_half
            double [:] al0_half = Ref.alpha0_half

        if np.isnan(var).any():
            print('plot var: ', message, ' NAN in variable ')

        plt.figure(1,figsize=(12,6))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        # plt.plot(var[th_varshift,:], Gr.z)
        plt.plot(var[w_varshift,:],Gr.z,'-x',label='var')
        plt.plot(M2.tendencies[w_varshift,w_varshift,:],Gr.z,label='M2.tend(ww)')
        plt.legend(fontsize=8)
        plt.title('ww')
        plt.subplot(1,4,2)
        plt.plot(M2.tendencies[w_varshift,w_varshift,:], Gr.z, label='M2.tend(ww)')
        plt.plot(M2.values[w_varshift,w_varshift,:],Gr.z,label='M2.ww')
        plt.plot(np.multiply(M1.values[w_varshift,:],M2.values[w_varshift,w_varshift,:]), Gr.z, label='M1.w*M2.ww')
        plt.plot(np.multiply(al0_half[:],np.multiply(M1.values[w_varshift,:],M2.values[w_varshift,w_varshift,:])), Gr.z, label='1/rho*M1.w*M2.ww')
        plt.plot(Ref.dz_rho0_half[Gr.gw:Gr.nzg-Gr.gw],Gr.z[Gr.gw:Gr.nzg-Gr.gw],label='dz rho')
        plt.legend(loc=4,fontsize=8)
        plt.title('ww')
        plt.subplot(1,4,3)
        plt.plot(var[w_varshift,Gr.gw:Gr.nzg-Gr.gw-1], Gr.z[Gr.gw:Gr.nzg-Gr.gw-1],'-x',label='var')
        plt.plot(M2.tendencies[w_varshift,w_varshift,Gr.gw:Gr.nzg-Gr.gw-1],Gr.z[Gr.gw:Gr.nzg-Gr.gw-1],label='M2.tend')
        plt.title('ww')
        plt.legend(fontsize=8)
        plt.subplot(1,4,4)
        # plt.plot(var[w_varshift,:], Gr.z)
        plt.plot(Ref.rho0_half,Gr.z,label='rho0_half')
        plt.title('aux')
        plt.legend(fontsize=8)
        plt.savefig('./figs/Turb_' + message + '_' + np.str(np.int(TS.t)) + '.png')
        plt.close()

        return




    def plot_var2(self, str message, n1,n2, var1, var2, Grid Gr, ReferenceState Ref, TimeStepping TS, MeanVariables M1, SecondOrderMomenta M2):
        cdef:
            Py_ssize_t th_varshift = M1.name_index['th']
            Py_ssize_t w_varshift = M1.name_index['w']
            Py_ssize_t v_varshift = M1.name_index['v']
            Py_ssize_t u_varshift = M1.name_index['u']
            double [:] rho0_half = Ref.rho0_half
            double [:] alpha0_half = Ref.alpha0_half
            Py_ssize_t gw = Gr.gw
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t nz = Gr.nz

        v1 = M1.index_name[n1]
        v2 = M1.index_name[n2]
        # print('plot_var2: ',  n1, v1, n2, v2)

        kmin = 1
        kmax = nzg-1

        plt.figure(1,figsize=(18,5))
        plt.subplot(1,5,1)
        plt.plot(Ref.alpha0_half[kmin:kmax],Gr.z[kmin:kmax],'-x',label='rho')
        plt.plot(Ref.alpha0_half[kmin:gw],Gr.z[kmin:gw], 'rx')
        plt.plot(Ref.alpha0_half[Gr.nz+gw:nzg],Gr.z[nz+gw:nzg],'rx')
        plt.title('alpha0_half')
        plt.subplot(1,5,2)
        plt.plot(M2.values[n1,n2,kmin:kmax],Gr.z[kmin:kmax],'-x',label='M2.'+ v1+v2)
        plt.plot(M2.values[n1,n2,kmin:gw],Gr.z[kmin:gw],'rx')
        plt.plot(M2.values[n1,n2,nz+gw:kmax],Gr.z[nz+gw:kmax],'rx')
        plt.title('M2.values ' + v1+v2+ ', '+ np.str(np.round(np.amax(M2.values[n1,n2,gw:nzg-gw]),6)), fontsize=10)
        plt.legend(fontsize=8)
        plt.subplot(1,5,3)
        plt.plot(M1.values[w_varshift,kmin:kmax],Gr.z[kmin:kmax],'-x',label='M1.w')
        plt.plot(M1.values[w_varshift,kmin:gw],Gr.z[kmin:gw],'rx')
        plt.plot(M1.values[w_varshift,nz+gw:kmax],Gr.z[nz+gw:kmax],'rx')
        plt.legend(fontsize=8)
        plt.title('M1.values w, '+np.str(np.round(np.amax(M1.values[w_varshift,gw:nzg-gw]),6)), fontsize=10)
        plt.subplot(1,5,4)
        plt.plot(var1[kmin:kmax],Gr.z[kmin:kmax],'-x',label='var1: '+v1+v2)
        plt.plot(var1[kmin:gw],Gr.z[kmin:gw],'rx')
        plt.plot(var1[nz+gw:kmax],Gr.z[nz+gw:kmax],'rx')
        plt.legend(fontsize=8)
        plt.title('var1 '+v1+v2+ ', '+ np.str(np.round(np.amax(var1[gw:nzg-gw]),6)), fontsize=10)
        plt.subplot(1,5,5)
        plt.plot(var2[kmin:kmax],Gr.z[kmin:kmax],'-x',label='var2:'+v1+v2)
        plt.plot(var2[kmin:Gr.gw],Gr.z[kmin:Gr.gw],'rx')
        plt.plot(var2[Gr.nz+Gr.gw:kmax],Gr.z[Gr.nz+Gr.gw:kmax],'rx')
        plt.legend(loc=4,fontsize=8)
        plt.title('var2 ' + v1+v2+ ', '+ np.str(np.round(np.amax(var2[gw:nzg-gw]),6)), fontsize=10)
        # plt.subplot(1,4,4)
        # # plt.plot(var[w_varshift,:], Gr.z)
        # plt.title('aux')
        # plt.legend(fontsize=8)
        plt.savefig('./figs/Turb_' + message + '_' + np.str(np.int(TS.t)) + '.png')
        plt.close()

        return



    cpdef plot_all(self, str message, Grid Gr, TimeStepping TS, MeanVariables M1, SecondOrderMomenta M2, double [:] var, int n1, int n2):
        cdef:
            Py_ssize_t th_index = M1.name_index['th']
            Py_ssize_t w_index = M1.name_index['w']
            Py_ssize_t v_index = M1.name_index['v']
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t gw = Gr.gw
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t nz = Gr.nz
            Py_ssize_t kmax

        if n1<=n2:
            v1 = M1.index_name[n1]
            v2 = M1.index_name[n2]
        else:
            v1 = M1.index_name[n2]
            v2 = M1.index_name[n1]
        # print('plot_var2: ',  n1, v1, n2, v2)

        if np.isnan(var).any():
            print('plot all: ', message, ' NAN in variable 1')

        kmin = 1
        kmax = nzg-1

        plt.figure(1,figsize=(20,5))
        plt.subplot(1,5,1)
        plt.plot(var[kmin:kmax],Gr.z[kmin:kmax],'-x',label='rho')
        plt.plot(var[kmin:gw],Gr.z[kmin:gw], 'rx')
        plt.plot(var[Gr.nz+gw:kmax],Gr.z[nz+gw:kmax],'rx')
        plt.title('Tend before Turb: '+v1+v2+ ', '+ np.str(np.round(np.amax(var[gw:nzg-gw]),7)), fontsize=12)
        plt.subplot(1,5,2)
        plt.plot(M2.tendencies[n1,n2,kmin:kmax],Gr.z[kmin:kmax],'-x',label='M2.tend: '+ v1+v2)
        plt.plot(M2.tendencies[n1,n2,kmin:gw],Gr.z[kmin:gw],'rx')
        plt.plot(M2.tendencies[n1,n2,nz+gw:kmax],Gr.z[nz+gw:kmax],'rx')
        plt.title('M2.tendency ' + v1+v2+ ', '+ np.str(np.round(np.amax(M2.tendencies[n1,n2,gw:nzg-gw]),7)), fontsize=12)
        plt.legend(fontsize=8)
        plt.subplot(1,5,3)
        plt.plot(M2.values[n1,n2,kmin:kmax],Gr.z[kmin:kmax],'k-x',label='M2.val: ' + v1+v2)
        plt.plot(M2.values[n1,n2,kmin:gw],Gr.z[kmin:gw],'rx')
        plt.plot(M2.values[n1,n2,nz+gw:kmax],Gr.z[nz+gw:kmax],'rx')
        plt.legend(fontsize=8)
        plt.title('M2.values , '+v1+v2+ ', '+np.str(np.round(np.amax(M2.values[n1,n2,gw:nzg-gw]),6)), fontsize=12)
        plt.subplot(1,5,4)
        if n1==2:
            n=n2
            v=v2
        else:
            n=n1
            v=v1
        plt.plot(M1.tendencies[n,kmin:kmax],Gr.z[kmin:kmax],'-x',label='M1.tend '+v)
        plt.plot(M1.tendencies[n,kmin:gw],Gr.z[kmin:gw],'rx')
        plt.plot(M1.tendencies[n,nz+gw:kmax],Gr.z[nz+gw:kmax],'rx')
        plt.legend(fontsize=8)
        plt.title('M1.tendency '+v+', '+np.str(np.round(np.amax(M1.tendencies[n,gw:nzg-gw]),6)), fontsize=12)
        plt.subplot(1,5,5)
        plt.plot(M1.values[n,kmin:kmax],Gr.z[kmin:kmax],'k-x',label='M1.'+v)
        plt.plot(M1.values[n,kmin:gw],Gr.z[kmin:gw],'rx')
        plt.plot(M1.values[n,nz+gw:kmax],Gr.z[nz+gw:kmax],'rx')
        plt.legend(fontsize=8)
        plt.title('M1.values '+v+', '+np.str(np.round(np.amax(M1.values[n,gw:nzg-gw]),6)), fontsize=12)
        plt.savefig('./figs/Turb_all_' + message + '_' + np.str(np.int(TS.t)) + '.pdf')
        plt.close()
        return


    cpdef plot_tendencies(self, str message, Grid Gr, TimeStepping TS, MeanVariables M1, SecondOrderMomenta M2):
        cdef:
            # double [:,:,:] tendencies = M2.tendencies
            double [:,:,:] tendencies = self.tendencies_M2
            Py_ssize_t th_varshift = M2.var_index['th']
            Py_ssize_t w_varshift = M2.var_index['w']
            Py_ssize_t v_varshift = M2.var_index['v']
            Py_ssize_t u_varshift = M2.var_index['u']

        if np.isnan(tendencies).any():
            print('!!!!!', message, ' NAN in M2 tendencies')
        plt.figure(2,figsize=(12,6))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(tendencies[th_varshift,th_varshift,:], Gr.z)
        plt.title('thth tend')
        plt.subplot(1,4,2)
        plt.plot(tendencies[w_varshift,w_varshift,:], Gr.z)
        plt.title('ww tend')
        plt.subplot(1,4,3)
        plt.plot(tendencies[w_varshift,u_varshift,:], Gr.z)
        plt.title('wu tend')
        plt.subplot(1,4,4)
        plt.plot(tendencies[w_varshift,th_varshift,:], Gr.z)
        plt.title('wth tend')
        # plt.show()
        plt.savefig('./figs/Turb_' + message + '_' + np.str(np.int(TS.t)) + '.png')
        plt.close()
        return
