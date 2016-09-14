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
        return

    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turbulence Base: update')
        return

    cpdef update_M1(self,Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
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
        print('Turb: update M1')
        cdef:
            Py_ssize_t k, m, n
            double [:] th0 = Ref.th0       # !!! right interfaces ???
            double dzi = Gr.dzi
            double [:] alpha0 = Ref.alpha0
            double [:] rho0_half = Ref.rho0_half

        n = M1.name_index['w']
        # (1) vertical edd fluxes: in all prognostic variables
        for var in M1.name_index.keys():
            turb = var + 'w'
            m = M1.name_index[var]
            for k in xrange(Gr.nzg):
                M1.tendencies[m,k] -= alpha0[k]*dzi*(rho0_half[k+1]*M2.values[m,n,k+1]-rho0_half[k]*M2.values[m,n,k])

        # (2) Buoyancy Flux: in w --> mean buoyancy approximated as zero (accurate if <rho> agrees with the reference state)
        # for n in xrange(M1.nv):
        #     for k in xrange(Gr.nzg):
        #         M1.tendencies[n,k] -= g/th0[k]*self.buoyancy[n,k]

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

    # cpdef initialize(self, Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef initialize(self, Grid Gr, PrognosticVariables.MeanVariables M1):
        print('Initializing Turbulence 2nd')
        self.buoyancy = np.zeros((M1.nv,Gr.nzg),dtype=np.double,order='c')
        return


    # cpdef update_M2(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
    cpdef update_M2(self, Grid Gr, ReferenceState Ref, TimeStepping TS, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turbulence 2nd order: update')
        # (1) Advection: MA.update_M2, SA.update_M2
        # (2) Diffusion: SD.update_M2, MD.update_M2
        # (3) Pressure: ?
        # (4) Third and higher order terms (first guess set to zero)
        M2.plot_tendencies('20',Gr,TS)
        self.advect_M2_local(Gr, Ref, M1, M2)
        M2.plot_tendencies('21',Gr,TS)
        # self.pressure_correlations_Mironov(Gr, M1, M2)
        M2.plot_tendencies('22',Gr,TS)
        self.pressure_correlations_Andre(Gr, M1, M2)
        M2.plot_tendencies('23',Gr,TS)
        self.pressure_correlations_Cheng(Gr, M1, M2)
        M2.plot_tendencies('24',Gr,TS)
        self.buoyancy_update(Gr, Ref, M1, M2)
        return



    cpdef buoyancy_update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        cdef:
            double [:] th0 = Ref.th0       # !!! right interfaces ???
            double [:] p0 = Ref.p0       # !!! right interfaces ???
            double [:] p = p0               # !!! need actual pressure !!!
            double [:] wql = np.zeros((Gr.nzg),dtype=np.double,order='c')       # !!!! how to get <w'ql'>????
            double [:,:] M2_b = np.zeros((M1.nv,Gr.nzg),dtype=np.double,order='c')

            int nth, nqt, m, k
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


        return


    cpdef advect_M2_local(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turb update: advect M2 local')
        # implemented for staggered grid

        cdef:
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            Py_ssize_t w_index = M1.name_index['w']
            Py_ssize_t th_index = M1.name_index['th']

            double [:] u = M1.values[u_index,:]
            double [:] v = M1.values[v_index,:]
            double [:] w = M1.values[w_index,:]
            double [:] th = M1.values[th_index,:]

            double [:] flux = np.zeros(Gr.nzg,dtype=np.double,order='c')
            double [:,:,:] tendencies = M2.tendencies
            double [:,:,:] values = M2.values
            double [:] rho0 = Ref.rho0
            double [:] alpha0_half = Ref.alpha0_half

            double dzi = Gr.dzi
            Py_ssize_t m, n, k, var_shift

        # (i) advection by mean vertical velocity
        # interpolate only M2 in fluxes (w on staggered grid)
        for m in xrange(M2.nv):
            for n in xrange(m,M2.nv):
                for k in xrange(1,Gr.nzg):
                    flux[k] = rho0[k]*w[k]*0.5*(M2.values[m,n,k]+M2.values[m,n,k-1])
                for k in xrange(1,Gr.nzg-1):
                    M2.tendencies[m,n,k] = -alpha0_half[k]*(flux[k+1]-flux[k])*dzi
                # if m==w_index and n==w_index:   # if name == 'ww':
                # # w on w-grid, M2.value on phi-grid --> compare to momentum advection for gradients
                #     for k in xrange(1,Gr.nzg-1):
                #         M2.tendencies[m,n,k] -= w[k]*(M2.values[m,n,k]-M2.values[m,n,k-1])*dzi
                # else:
                # # w and M2.value both on w-grid --> compare to scalar advection for gradients
                #     for k in xrange(Gr.nzg):
                #         M2.tendencies[m,n,k] -= 0.5*(w[k]+w[k+1])*(M2.values[m,n,k]-M2.values[m,n,k-1])*dzi

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
            tendencies[u_index, u_index, k] = -2*values[u_index,w_index,k] * (u[k+1]-u[k])*dzi
            tendencies[v_index, v_index, k] = -2*values[v_index,w_index,k] * (v[k+1]-v[k])*dzi
            tendencies[u_index, w_index, k] = -values[w_index,w_index,k]*(u[k+1]-u[k])*dzi - values[u_index,w_index,k]*(w[k+1]-w[k])*dzi
            tendencies[v_index, w_index, k] = -values[w_index,w_index,k]*(v[k+1]-v[k])*dzi - values[v_index,w_index,k]*(w[k+1]-w[k])*dzi
            tendencies[w_index, w_index, k] = -2*values[w_index,w_index,k]*(w[k+1]-w[k])*dzi
            tendencies[u_index, th_index, k] = -values[w_index,th_index,k]*(u[k+1]-u[k])*dzi - values[u_index,w_index,k]*(th[k+1]-th[k])*dzi
            tendencies[v_index, th_index, k] = -values[w_index,th_index,k]*(v[k+1]-v[k])*dzi - values[v_index,w_index,k]*(th[k+1]-th[k])*dzi
            tendencies[w_index, th_index, k] = -values[w_index,th_index,k]*(w[k+1]-w[k])*dzi - values[w_index,w_index,k]*(th[k+1]-th[k])*dzi
            tendencies[th_index, th_index, k] = -2*values[w_index,th_index,k]*(th[k+1]-th[k])*dzi

        if 'qt' in M1.index_name:
            qt_index = M1.name_index['qt']
            qt = M1.values[qt_index,:]
            tendencies[u_index, qt_index, k] = -values[w_index,qt_index,k]*(u[k+1]-u[k])*dzi - values[u_index,w_index,k]*(qt[k+1]-qt[k])*dzi
            tendencies[v_index, qt_index, k] = -values[w_index,qt_index,k]*(v[k+1]-v[k])*dzi - values[v_index,w_index,k]*(qt[k+1]-qt[k])*dzi
            tendencies[w_index, qt_index, k] = -values[w_index,qt_index,k]*(w[k+1]-w[k])*dzi - values[w_index,w_index,k]*(qt[k+1]-qt[k])*dzi
            tendencies[qt_index, qt_index, k] = -2*values[w_index,qt_index,k]*(qt[k+1]-qt[k])*dzi

        return



    cpdef pressure_correlations_Mironov(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        '''following Mironov (2009), based on André (1978), Launder (1975) and Rotta (1951)'''
        cdef:
        #     Py_ssize_t uu_shift = M2.get_varshift(Gr, 'uu')
        #     Py_ssize_t vv_shift = M2.get_varshift(Gr, 'vv')
        #     Py_ssize_t wu_shift = M2.get_varshift(Gr, 'wu')
        #     Py_ssize_t wv_shift = M2.get_varshift(Gr, 'wv')
        #     Py_ssize_t ww_shift = M2.get_varshift(Gr, 'ww')
        #     Py_ssize_t pu_shift = M2.get_varshift(Gr, 'pu')
        #     Py_ssize_t pv_shift = M2.get_varshift(Gr, 'pv')
        #     Py_ssize_t pw_shift = M2.get_varshift(Gr, 'pw')
        #     Py_ssize_t uth_shift = M2.get_varshift(Gr, 'uth')
        #     Py_ssize_t vth_shift = M2.get_varshift(Gr, 'vth')
        #     Py_ssize_t wth_shift = M2.get_varshift(Gr, 'wth')
        #
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


    cpdef pressure_correlations_Andre(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        # '''following André (1978), based on Launder (1975) and Rotta (1951)'''
        # !!! only dry thermodynamics !!!
        # !!! only expression for vertical moisture flux (no horizontal fluxes) !!!
        print('Turb update: pressure correlations André')

        time = 0
        # self.plot('30', Gr, time, M1, M2)
        # self.plot_tendencies('30', Gr, time, M1, M2)

        cdef:
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            Py_ssize_t w_index = M1.name_index['w']
            Py_ssize_t th_index = M1.name_index['th']

            # double [:] u = M1.values[u_index,:]
            # double [:] v = M1.values[v_index,:]

            double [:,:,:] P = np.zeros(shape=(M1.nv,M1.nv,Gr.nzg),dtype=np.double,order='c')
            double [:] P_tke = np.zeros((Gr.nzg),dtype=np.double,order='c')
            double [:] tke = np.zeros((Gr.nzg),dtype=np.double,order='c')

            double lb=0.0
            double ld=0.0
            double l0, karman, c1
            double [:] l = np.zeros((Gr.nzg),dtype=np.double,order='c')
            double [:] epsilon = np.zeros((Gr.nzg),dtype=np.double,order='c')

            Py_ssize_t i,j,k
            double dzi2 = 0.5*Gr.dzi
            double dzi = Gr.dzi

            double alpha = 1/285    # thermal expansion coefficient
            double c4 = 4.5
            double c5 = 0.0
            double c6 = 4.85
            double c7 = 1.0-0.125*c6


        # (1) TKE and TKE generation rate
        for k in xrange(Gr.nzg):
            # Kinetic Energy ---> Diagnostic Variable (needed elsewhere?!?!)
            tke[k] = np.sqrt(M2.values[0,0,k] + M2.values[1,1,k] + M2.values[2,2,k])
            # TKE generation rate
            P_tke[k] = alpha*g*M2.values[w_index,th_index,k] \
                       - M2.values[u_index,w_index,k]*(M1.values[u_index,k+1]-M1.values[u_index,k])*dzi \
                       - M2.values[v_index,w_index,k]*(M1.values[v_index,k+1]-M1.values[v_index,k])*dzi


        # (2) Energy Dissipation
        # lb: Mixing length  for neutral and unstable conditions (Blackadar, 1962)
        # ld: characteristic lengthfor stable stratification
        karman = 0.35           # von Karman constant
        l0 = 15           # mixing length far above the ground, take 15m (Jieliu, 2011)
        if np.amin(tke)>0:
            for k in xrange(Gr.nzg-1):
                lb = karman*k/(1+karman*k/l0)
                ld = 0.75/(np.sqrt(tke[k]))*1/np.sqrt(alpha*g*(M1.values[th_index,k+1]-M1.values[th_index,k])*dzi)
                l[k] = np.min(lb,ld)
                c1 = 0.019+0.051*l[k]/lb
                epsilon[k] = c1/np.sqrt(tke[k]*tke[k]*tke[k])


        # (3) Momentum Fluxes: generation rate and tendency
        for i in xrange(M1.nv_velocities):
            for k in xrange(Gr.nzg):
                P[i,w_index,k] = alpha*g*(M2.values[i,th_index,k])
                P[w_index,i,k] = alpha*g*(M2.values[th_index,i,k])
            for k in xrange(Gr.nzg-1):
                M2.tendencies[i,i,k] += 2/3*( c4*epsilon[k] + c5*P_tke[k])
        # self.plot_tendencies('31', Gr, time, M1, M2)

        for i in xrange(M1.nv_velocities):
            for j in xrange(i,M1.nv_velocities):
                for k in xrange(Gr.nzg-1):
                    P[i,j,k] -= M2.values[i,w_index,k]*(M1.values[j,k+1]-M1.values[j,k])*dzi \
                                + M2.values[j,w_index,k]*(M1.values[i,k+1]-M1.values[i,k])*dzi
                    if np.isnan(P[i,j,k]):
                        print('....... P is nan, ', i,j,k)
                for k in xrange(Gr.nzg-1):
                    # M2.tendencies[i,j,k] += -c4*epsilon[k]/tke[k]*M2.values[i,j,k] - c5*P[i,j,k]
                    if tke[k] > 0:
                        M2.tendencies[i,j,k] += -c4*epsilon[k]/tke[k]*M2.values[i,j,k] - c5*P[i,j,k]
                    else:
                        M2.tendencies[i,j,k] -= c5*P[i,j,k]     # c5=0

        print('epsilon', np.isnan(epsilon).any(), np.isnan(M2.values).any(), np.isnan(P).any())
        print('tke', np.isnan(tke).any(), np.amin(tke), np.amax(tke))

        # self.plot_tendencies('32', Gr, time, M1, M2)
        for i in xrange(M1.nv_velocities):
        # (4) Heat Flux generation rate and tendency
            for k in xrange(Gr.nzg-1):
                P[i,th_index,k] = alpha*g*M2.values[w_index,th_index,k] \
                                  - M2.values[u_index,w_index,k]*(M1.values[u_index,k+1]-M1.values[u_index,k])*dzi \
                                  - M2.values[v_index,w_index,k]*(M1.values[v_index,k+1]-M1.values[v_index,k])*dzi
            for k in xrange(Gr.nzg-1):
                if tke[k] > 0:
                    M2.tendencies[i,th_index,k] += -c6*epsilon[k]/tke[k]*M2.values[i,th_index,k] - c7*P[i,th_index,k]
                else:
                    M2.tendencies[i,th_index,k] -= c7*P[i,th_index,k]

        self.plot_tendencies('33', Gr, time, M1, M2)

        # (5) Moisture Flux generation rate and tendency
        if 'qt' in M1.name_index:
            qt_index = M1.name_index['qt']

            for k in xrange(Gr.nzg-1):
                if th_index < qt_index:
                    P[i,qt_index,k] = alpha*g*M2.values[th_index,qt_index,k]
                else:
                    P[i,qt_index,k] = alpha*g*M2.values[qt_index,th_index,k]
            for k in xrange(Gr.nzg-1):
                M2.tendencies[w_index,qt_index,k] = -c6*epsilon[k]/tke[k]*M2.values[w_index,qt_index,k] \
                                                    - c7*P[w_index,qt_index,k]

        self.plot_tendencies('34', Gr, time, M1, M2)

        time += 1
        return


    cpdef pressure_correlations_Cheng(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        cdef:
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            Py_ssize_t w_index = M1.name_index['w']


        # for k in xrange(Gr.nzg):
        #     Kinetic Energy ---> Diagnostic Variable (needed elsewhere?!?!)
            # tke[k] = np.sqrt(M2.values[0,0,k] + M2.values[1,1,k] + M2.values[2,2,k])
        return


    # cpdef stats_io(self, Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
    #              PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return

    cpdef plot(self, str message, Grid Gr, int time, MeanVariables M1, SecondOrderMomenta M2):
        cdef:
            double [:,:,:] values = M2.values
            Py_ssize_t th_varshift = M1.name_index['th']
            Py_ssize_t w_varshift = M1.name_index['w']
            Py_ssize_t v_varshift = M1.name_index['v']
            Py_ssize_t u_varshift = M1.name_index['u']

        if np.isnan(values).any():
            print('!!!!!', message, ' NAN in M2 tendencies')

        plt.figure(1,figsize=(15,7))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(values[th_varshift,th_varshift,:], Gr.z)
        plt.title('thth')
        plt.subplot(1,4,2)
        plt.plot(values[w_varshift,w_varshift,:], Gr.z)
        plt.title('ww')
        plt.subplot(1,4,3)
        plt.plot(values[w_varshift,u_varshift,:], Gr.z)
        plt.title('wu')
        plt.subplot(1,4,4)
        plt.plot(values[w_varshift,th_varshift,:], Gr.z)
        plt.title('wth')
        # plt.show()
        plt.savefig('./figs/M2_profiles_' + message + '_' + np.str(time) + '.png')
        plt.close()


    cpdef plot_tendencies(self, str message, Grid Gr, int time, MeanVariables M1, SecondOrderMomenta M2):
        cdef:
            double [:,:,:] tendencies = M2.tendencies
            Py_ssize_t th_varshift = M2.var_index['th']
            Py_ssize_t w_varshift = M2.var_index['w']
            Py_ssize_t v_varshift = M2.var_index['v']
            Py_ssize_t u_varshift = M2.var_index['u']

        if np.isnan(tendencies).any():
            print('!!!!!', message, ' NAN in M2 tendencies')
        plt.figure(2,figsize=(15,7))
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
        plt.savefig('./figs/M2_tendencies_' + message + '_' + np.str(time) + '.png')
        plt.close()
        return
