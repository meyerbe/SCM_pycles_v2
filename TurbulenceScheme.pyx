#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from Grid cimport Grid
cimport PrognosticVariables
from ReferenceState cimport ReferenceState
# cimport DiagnosticVariables
# cimport Kinematics
# cimport Surface
# from NetCDFIO cimport NetCDFIO_Stats
from libc.math cimport exp, sqrt
cimport numpy as np
import numpy as np
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

        n = M1.name_index['w']
        # (1) vertical eddy fluxes: in all prognostic variables
        for var in M1.name_index.keys():
            turb = var + 'w'
            m = M1.name_index[var]
            for k in xrange(Gr.nzg):
                M1.tendencies[m,k] -= M2.values[m,n,k]

        # (2) Buoyancy Flux: in w
        for n in xrange(M1.nv):
            for k in xrange(Gr.nzg):
                M1.tendencies[n,k] -= g/th0[k]*self.buoyancy[n,k]

        # (3) Pressure Correlations
        '''w'''
        n = M1.name_index['w']
        # for k in xrange(Gr.nzg):
        #     M1.tendencies[n,k] -= M1.tendencies[]
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


    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        print('Turbulence 2nd order: update')
        # (1) Advection: MA.update_M2, SA.update_M2
        # (2) Diffusion: SD.update_M2, MD.update_M2
        # (3) Pressure: ?
        # (4) Third and higher order terms (first guess set to zero)
        # self.update_M1(Gr, Ref, M1, M2)
        self.advect_M2_local(Gr, M1, M2)
        self.pressure_correlations_Mironov(Gr, M1, M2)
        self.pressure_correlations_Andre(Gr, M1, M2)
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

        # (2) Buoyancy Flux: in w
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
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            Py_ssize_t w_index = M1.name_index['w']
            Py_ssize_t th_index = M1.name_index['th']

        #     Py_ssize_t wu_shift = M2.get_varshift(Gr, 'wu')
        #     Py_ssize_t wv_shift = M2.get_varshift(Gr, 'wv')
        #     Py_ssize_t ww_shift = M2.get_varshift(Gr, 'ww')
        #     Py_ssize_t wth_shift = M2.get_varshift(Gr, 'wth')

            double [:] u = M1.values[u_index,:]
            double [:] v = M1.values[v_index,:]
            double [:] w = M1.values[w_index,:]
            double [:] th = M1.values[th_index,:]

            double dzi = Gr.dzi
            Py_ssize_t m, n, k, var_shift


        # (i) advection by mean vertical velocity
        for m in xrange(M1.nv):
            for n in xrange(m,M1.nv):
                if m==w_index and n==w_index:   # if name != 'ww':
                # w on w-grid, M2.value on phi-grid --> compare to momentum advection for gradients
                    for k in xrange(1,Gr.nzg-1):
                        M2.tendencies[m,n,k] -= w[k]*(M2.values[m,n,k]-M2.values[m,n,k-1])*dzi
                else:
                # w and M2.value both on w-grid --> compare to scalar advection for gradients
                    for k in xrange(Gr.nzg):
                        M2.tendencies[m,n,k] -= 0.5*(w[k]+w[k+1])*(M2.values[m,n,k]-M2.values[m,n,k-1])*dzi


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
            M2.tendencies[u_index,w_index,k] -= 0.5*(M2.values[w_index,w_index,k]+M2.values[w_index,w_index,k+1])*(u[k]-u[k-1])*dzi \
                                         - M2.values[u_index,w_index,k]*( 0.5*(w[k]+w[k+1]) - 0.5*(w[k-1]+w[k]) )*dzi
            M2.tendencies[v_index,w_index,k] -= 0.5*(M2.values[w_index,w_index,k]+M2.values[w_index,w_index,k+1])*(v[k]-v[k-1])*dzi \
                                         - M2.values[v_index,w_index,k]*( 0.5*(w[k]+w[k+1]) - 0.5*(w[k-1]+w[k]) )*dzi
            M2.tendencies[w_index,w_index,k] -= M2.values[w_index,w_index,k]*(w[k]-w[k-1])*dzi
            M2.tendencies[w_index,th_index,k] -= 0.5*(M2.values[w_index,w_index,k]+M2.values[w_index,w_index,k+1])*(th[k]-th[k-1])*dzi \
                                         - M2.values[w_index,th_index,k]*( 0.5*(w[k]+w[k+1]) - 0.5*(w[k-1]+w[k]) )*dzi
            # M2.tendencies[wu_shift,k] -= 0.5*(M2.values[ww_shift,k]+M2.values[ww_shift,k+1])*(u[k]-u[k-1])*dzi \
            #                              - M2.values[wu_shift,k]*( 0.5*(w[k]+w[k+1]) - 0.5*(w[k-1]+w[k]) )*dzi
            # M2.tendencies[wv_shift,k] -= 0.5*(M2.values[ww_shift,k]+M2.values[ww_shift,k+1])*(v[k]-v[k-1])*dzi \
            #                              - M2.values[wv_shift,k]*( 0.5*(w[k]+w[k+1]) - 0.5*(w[k-1]+w[k]) )*dzi
            # M2.tendencies[ww_shift,k] -= M2.values[ww_shift,k]*(w[k]-w[k-1])*dzi
            # M2.tendencies[wth_shift,k] -= 0.5*(M2.values[ww_shift,k]+M2.values[ww_shift,k+1])*(s[k]-s[k-1])*dzi \
            #                              - M2.values[wth_shift,k]*( 0.5*(w[k]+w[k+1]) - 0.5*(w[k-1]+w[k]) )*dzi

        return

    cpdef pressure_correlations_Mironov(self, Grid Gr, PrognosticVariables.MeanVariables M1, PrognosticVariables.SecondOrderMomenta M2):
        '''following Mironov (2009), based on André (1978), Launder (1975) and Rotta (1951)'''
        # cdef:
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
        #     Py_ssize_t u_shift = M1.get_varshift(Gr, 'u')
        #     Py_ssize_t v_shift = M1.get_varshift(Gr, 'v')
        #     Py_ssize_t w_shift = M1.get_varshift(Gr, 'w')
        #
        #     Py_ssize_t k, shift
        #     double dzi2 = 0.5*Gr.dzi
        #
        #     double [:] tke = np.zeros((Gr.nzg),dtype=np.double,order='c')
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
        #
        #
        # # M2.name_index:    {'pv': 6, 'pw': 7, 'pu': 5, 'uu': 0, 'vth': 9, 'uth': 8, 'ww': 4, 'wv': 3, 'wu': 2, 'vv': 1, 'wth': 10}
        # # M2.name_index.keys():     gives only names (=keys)
        # # M2.name_index['uu']:      gives the index of variable with name (or key) 'uu'
        # # M2.index_name[0]:         gives name (or key) at position 0
        #
        # for k in xrange(Gr.nzg):
        #     # Kinetic Energy ---> Diagnostic Variable (needed elsewhere?!?!)
        #     tke[k] = np.sqrt(M2.values[uu_shift,k] + M2.values[vv_shift,k] + M2.values[ww_shift,k])
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
        # '''following Mironov (2009), based on André (1978), Launder (1975) and Rotta (1951)'''
        # cdef:
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
        #     Py_ssize_t u_shift = M1.get_varshift(Gr, 'u')
        #     Py_ssize_t v_shift = M1.get_varshift(Gr, 'v')
        #     Py_ssize_t w_shift = M1.get_varshift(Gr, 'w')
        #
        #     Py_ssize_t k, shift
        #     double dzi2 = 0.5*Gr.dzi
        #
        #     double [:] tke = np.zeros((Gr.nzg),dtype=np.double,order='c')
        #
        #     double c4 = 4.5
        #     double c5 = 0.0
        #     double c6 = 4.85
        #     double c7 = 1.0-0.125*c6
        #
        # for k in xrange(Gr.nzg):
        #     # Kinetic Energy ---> Diagnostic Variable (needed elsewhere?!?!)
        #     tke[k] = np.sqrt(M2.values[uu_shift,k] + M2.values[vv_shift,k] + M2.values[ww_shift,k])
        #     # puu[k] =
        #
        # var = 'uu'
        # index = M2.name_index[var]          # get index of var
        # shift = M2.get_varshift(Gr, var)
        # for k in xrange(1,Gr.nzg-1):
        #     # departure-from-isotropy tensor
        #     a = 2*M2.values[shift,k]/tke[k] - 2.0/3.0
        #     # Mean Shear ---> Diagnostic Variable (needed elsewhere?!?!)
        #     Sxz = 0.5*( (M1.values[u_shift,k+1]-M1.values[u_shift,k-1])*dzi2 )
        #     Wzx = 0.5*( (M1.values[u_shift,k+1]-M1.values[u_shift,k-1])*dzi2 )
        #     Wxz = -Wzx
        #
        #     tend = 0.0
        #
        #     M2.tendencies[wu_shift,k] += 0
        #     M2.tendencies[wu_shift,k] += 0
        #     M2.tendencies[wu_shift,k] += 0
        #     M2.tendencies[wu_shift,k] += 0
        #     M2.tendencies[wu_shift,k] += 0
        #     M2.tendencies[wu_shift,k] += 0

        return



    # cpdef stats_io(self, Grid Gr,  DiagnosticVariables.DiagnosticVariables DV,
    #              PrognosticVariables.PrognosticVariables PV, Kinematics.Kinematics Ke, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return

