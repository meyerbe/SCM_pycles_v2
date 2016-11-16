#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from Grid cimport Grid
from ReferenceState cimport ReferenceState
from PrognosticVariables cimport PrognosticVariables
from PrognosticVariables cimport MeanVariables
from PrognosticVariables cimport SecondOrderMomenta
# cimport DiagnosticVariables
cimport TimeStepping
# cimport Radiation
# from Thermodynamics cimport LatentHeat,ClausiusClapeyron
# from SurfaceBudget cimport SurfaceBudget
from NetCDFIO cimport NetCDFIO_Stats


from thermodynamic_functions cimport cpm_c, pv_c, pd_c, exner
from surface_functions cimport compute_ustar

from libc.math cimport sqrt, log, fabs,atan, exp, fmax
import cython
cimport numpy as np
import numpy as np
include "parameters.pxi"


# cdef extern from "thermodynamic_functions.h":
#     inline double theta_rho_c(double p0, double T,double qt, double qv) nogil
# cdef extern from "surface.h":
#     double compute_ustar(double windspeed, double buoyancy_flux, double z0, double z1) nogil
#     inline double entropyflux_from_thetaflux_qtflux(double thetaflux, double qtflux, double p0_b, double T_b, double qt_b, double qv_b) nogil
#     void compute_windspeed(Grid.DimStruct *dims, double* u, double*  v, double*  speed, double u0, double v0, double gustiness ) nogil
#     void exchange_coefficients_byun(double Ri, double zb, double z0, double* cm, double* ch, double* lmo) nogil
# cdef extern from "entropies.h":
#     inline double sd_c(double pd, double T) nogil
#     inline double sv_c(double pv, double T) nogil

# def SurfaceFactory(namelist, LatentHeat LH):
def SurfaceFactory(namelist):
        try:
            casename = namelist['meta']['casename']
        except:
            casename = 'None'
        if casename == 'SullivanPatton':
            # return SurfaceSullivanPatton(LH)
            return SurfaceSullivanPatton()
        # elif casename == 'Bomex':
        #     return SurfaceBomex(LH)
        # elif casename == 'Gabls':
        #     return SurfaceGabls(namelist,LH)
        # elif casename == 'DYCOMS_RF01':
        #     return SurfaceDYCOMS_RF01(namelist, LH)
        # elif casename == 'DYCOMS_RF02':
        #     return SurfaceDYCOMS_RF02(namelist, LH)
        # elif casename == 'Rico':
        #     return SurfaceRico(LH)
        # elif casename == 'CGILS':
        #     return SurfaceCGILS(namelist, LH, Par)
        # elif casename == 'ZGILS':
        #     return SurfaceZGILS(namelist, LH, Par)
        # elif casename == 'DCBLSoares':
        if casename == 'DCBLSoares':
            # return SurfaceSoares(LH)
            return SurfaceSoares()
        # elif casename == 'DCBLSoares_moist':
        #     return SurfaceSoares_moist(LH)
        else:
            return SurfaceNone()



cdef class SurfaceBase:
    def __init__(self):
        return
    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
        self.u_flux = 0.0
        self.v_flux = 0.0
        self.qt_flux = 0.0
        self.th_flux = 0.0

        self.obukhov_length = 0.0
        self.friction_velocity = 0.0
        self.shf = 0.0
        self.lhf = 0.0
        self.b_flux = 0.0

        # If not overridden in the specific case, set T_surface = Tg
        self.T_surface = Ref.Tg

        NS.add_ts('uw_surface', Gr)
        NS.add_ts('vw_surface', Gr)
        NS.add_ts('s_flux_surface', Gr)
        NS.add_ts('shf_surface', Gr)
        NS.add_ts('lhf_surface', Gr)
        NS.add_ts('obukhov_length', Gr)
        NS.add_ts('friction_velocity', Gr)
        NS.add_ts('buoyancy_flux_surface', Gr)

        return

    # cpdef init_from_restart(self, Restart):
    #     self.T_surface = Restart.restart_data['surf']['T_surf']
    #     return
    # cpdef restart(self, Restart):
    #     Restart.restart_data['surf'] = {}
    #     Restart.restart_data['surf']['T_surf'] = self.T_surface
    #     return
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, SecondOrderMomenta M2, TimeStepping.TimeStepping TS):
        # DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
        cdef :
            Py_ssize_t gw = Gr.gw
            # Py_ssize_t t_index = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t th_index = M1.name_index['th']
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            Py_ssize_t w_index = M1.name_index['w']
            Py_ssize_t ql_index, qt_index
            double [:] t_mean = np.zeros(Gr.nzg) #=  Pa.HorizontalMean(Gr, &DV.values[t_shift])
            # double cp_, lam, lv, pv, pd
            # double sv, sd
            # double dzi = Gr.dzi
            # double tendency_factor = Ref.alpha0_half[gw]/Ref.alpha0[gw-1]*dzi

        if self.dry_case:
            temperature = 293.0
            t_mean[gw] = temperature
            print('!!! not correct temperature in Surface Base update !!!')
            self.shf = self.s_flux * Ref.rho0_half[gw] * temperature#DV.values[t_index,gw]
            self.b_flux = self.shf * g * Ref.alpha0_half[gw]/cpd/t_mean[gw]
            self.obukhov_length = -self.friction_velocity * self.friction_velocity * self.friction_velocity /self.b_flux/vkb

            # PV.tendencies[u_index,gw] +=  self.u_flux * tendency_factor
            # PV.tendencies[v_index,gw] +=  self.v_flux * tendency_factor
            # PV.tendencies[th_index,gw] +=  self.th_flux * tendency_factor
            M2.tendencies[u_index,w_index,gw] +=  self.u_flux
            M2.tendencies[v_index,w_index,gw] +=  self.v_flux
            M2.tendencies[w_index,th_index,gw] +=  self.th_flux

        else:
            print('Surface Base: get DV ql')
            # ql_index = DV.get_varshift(Gr,'ql')
            qt_index = M1.name_index['qt']
            temperature = 293.0
            t_mean[gw] = temperature
            lam = 1.0
            lv = 1.0
            print('!!! not correct temperature, ql, latent heat, lambda in Surface Base update !!!')
            print('!!! not correct moist thermodynamics (potential temperature) in Surface Base !!!')
            # lam = self.Lambda_fp(DV.values[t_index,gw])
            # lv = self.L_fp(DV.values[t_index,gw],lam)
            self.lhf = self.qt_flux * Ref.rho0_half[gw] * lv
            # pv = pv_c(Ref.p0_half[gw], PV.values[qt_index,gw], PV.values[qt_index,gw] - DV.values[ql_index,gw])
            # pd = pd_c(Ref.p0_half[gw], PV.values[qt_index,gw], PV.values[qt_index,gw] - DV.values[ql_index,gw])
            # sv = sv_c(pv,DV.values[t_index,gw])
            # sd = sd_c(pd,DV.values[t_index,gw])
            print('!!! Surface Base: compute liquid pot. temperature instead of entropy')
            # self.shf = (self.th_flux * Ref.rho0_half[gw] - self.lhf/lv * (sv-sd)) * DV.values[t_index,gw]
            # cp_ = cpm_c(PV.values[qt_index,gw])
            # self.b_flux = g * Ref.alpha0_half[gw]/cp_/t_mean[gw] * \
            #               (self.shf + (eps_vi-1.0)*cp_*t_mean[gw]*self.lhf/lv)
            # self.obukhov_length = -self.friction_velocity *self.friction_velocity *self.friction_velocity / self.b_flux/vkb

            # PV.tendencies[u_index,gw] +=  self.u_flux * tendency_factor
            # PV.tendencies[v_index,gw] +=  self.v_flux * tendency_factor
            # PV.tendencies[th_index,gw] +=  self.th_flux * tendency_factor
            # PV.tendencies[qt_index,gw] +=  self.qt_flux * tendency_factor
            M2.tendencies[u_index,w_index,gw] +=  self.u_flux
            M2.tendencies[v_index,w_index,gw] +=  self.v_flux
            M2.tendencies[w_index,th_index,gw] +=  self.th_flux
            M2.tendencies[w_index,qt_index,gw] +=  self.qt_flux

        return

    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
        NS.write_ts('uw_surface', self.u_flux)
        NS.write_ts('vw_surface', self.v_flux)
        NS.write_ts('th_flux_surface', self.th_flux)
        NS.write_ts('buoyancy_flux_surface', self.b_flux)
        NS.write_ts('shf_surface', self.shf)
        NS.write_ts('lhf_surface', self.lhf)

        NS.write_ts('friction_velocity', self.friction_velocity)
        NS.write_ts('obukhov_length', self.obukhov_length)
        return




cdef class SurfaceNone(SurfaceBase):
    def __init__(self):
        pass
    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
        return
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, SecondOrderMomenta M2, TimeStepping.TimeStepping TS):
    #                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
        return
    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
        return



'''SULLIVAN'''
cdef class SurfaceSullivanPatton(SurfaceBase):
    def __init__(self):
        self.theta_flux = 0.24  # K m/s
        self.z0 = 0.1           #m (Roughness length)
        self.gustiness = 0.001  #m/s, minimum surface windspeed for determination of u*
        self.dry_case = True
        return

    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
        T0 = Ref.p0_half[Gr.gw] * Ref.alpha0_half[Gr.gw]/Rd
        self.buoyancy_flux = self.theta_flux * exner(Ref.p0_half[Gr.gw]) * g /T0
        SurfaceBase.initialize(self,Gr,Ref,NS)
        return

    # cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV,
    #              DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, SecondOrderMomenta M2, TimeStepping.TimeStepping TS):
        # Since this case is completely dry, the liquid potential temperature is equivalent to the potential temperature
        cdef:
            Py_ssize_t gw = Gr.gw
            # Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
        print('!!! Surface Scheme Sullivan: wrong temperature !!!')

        # no computation necessary, since theta-flux directly given and dry dynamics
        # self.s_flux = cpd * self.theta_flux*exner(Ref.p0_half[gw])/DV.values[temp_shift,gw]
        # temperature = 293.0
        # self.s_flux = cpd * self.theta_flux*exner(Ref.p0_half[gw])/temperature

        cdef:
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            double windspeed = 0.0
            double u = M1.values[u_index,gw] + Ref.u0
            double v = M1.values[v_index,gw] + Ref.v0
        # compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed,Ref.u0, Ref.v0,self.gustiness)
        windspeed = np.fmax(np.sqrt(u*u + v*v), self.gustiness)

        # Get the shear stresses
        self.friction_velocity = compute_ustar(windspeed,self.buoyancy_flux, self.z0, Gr.dz/2.0)
        self.u_flux = -self.friction_velocity**2 / windspeed * (PV.values[u_index,gw] + Ref.u0)
        self.v_flux = -self.friction_velocity**2 / windspeed * (PV.values[v_index,gw] + Ref.v0)

        SurfaceBase.update(self, Gr, Ref, PV, M1, M2, TS)
        return

    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
        SurfaceBase.stats_io(self, Gr, NS)
        return



'''SOARES'''
# like in Sullivan case: z0 is given (ustar_fixed = 'False')
# like in Bomex case: surface heat and moisture flux constant and prescribed
cdef class SurfaceSoares(SurfaceBase):
    def __init__(self):
        self.z0 = 0.001 #m (Roughness length)
        self.gustiness = 0.001 #m/s, minimum surface windspeed for determination of u*
        return

    # @cython.boundscheck(False)  #Turn off numpy array index bounds checking
    # @cython.wraparound(False)   #Turn off numpy array wrap around indexing
    # @cython.cdivision(True)
    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):     # Sullivan
        SurfaceBase.initialize(self,Gr,Ref,NS)

        self.theta_surface = 300.0 # K
        # self.qt_surface = 5.0e-3 # kg/kg

        self.theta_flux = 0.06 # K m/s
        # self.qt_flux = 2.5e-5 # m/s

        # Bomex:
        # self.buoyancy_flux = g * ((self.theta_flux + (eps_vi-1.0)*(self.theta_surface*self.qt_flux[ij] + self.qt_surface *self.theta_flux))
        #                       /(self.theta_surface*(1.0 + (eps_vi-1)*self.qt_surface)))     # adopted from Bomex ??
        # Sullivan:
        T0 = Ref.p0_half[Gr.gw] * Ref.alpha0_half[Gr.gw]/Rd
        self.buoyancy_flux = self.theta_flux * exner(Ref.p0_half[Gr.gw]) * g /T0

        return

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    # update adopted and modified from Sullivan + Bomex
    # cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, SecondOrderMomenta M2, TimeStepping.TimeStepping TS):
        # Since this case is completely dry, the computation of entropy flux from sensible heat flux is very simple
        cdef:
            Py_ssize_t gw = Gr.gw
            # Py_ssize_t imax = Gr.dims.nlg[0]
            # Py_ssize_t jmax = Gr.dims.nlg[1]
            # Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            # Py_ssize_t jstride = Gr.dims.nlg[2]
            # Py_ssize_t istride_2d = Gr.dims.nlg[1]
            # Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t th_index = M1.name_index['th']
            Py_ssize_t qt_index, qv_index

        # Scalar fluxes (adopted from Bomex)
        temperature = 293.0
        print('!!! Surface Base Soares: wrong temperature, surface flux s_flux --> th_flux !!!')
        # Sullivan
        # self.s_flux = cpd * self.theta_flux*exner(Ref.p0_half[gw])/DV.values[temp_shift,gw]
        self.s_flux = cpd * self.theta_flux*exner(Ref.p0_half[gw])/temperature
        # Bomex (entropy flux includes qt flux)
        # self.s_flux = entropyflux_from_thetaflux_qtflux(self.theta_flux, self.qt_flux, Ref.p0_half[gw], DV.values[temp_shift,gw], PV.values[qt_shift,gw], DV.values[qv_shift,gw])

        # Windspeed (adopted from Sullivan, equivalent to Bomex)
        cdef:
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            double windspeed = 0.0
            double u = M1.values[u_index,gw] + Ref.u0
            double v = M1.values[v_index,gw] + Ref.v0
        # compute_windspeed(&Gr.dims, &PV.values[u_index], &PV.values[v_index], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)
        windspeed = np.fmax(np.sqrt(u*u + v*v), self.gustiness)

       # Surface Values: friction velocity, obukhov lenght (adopted from Sullivan, since same Surface parameters prescribed)
        print('Surface Base Soares: compute friction velocity !!!')
       #  self.friction_velocity = compute_ustar(windspeed,self.buoyancy_flux,self.z0, Gr.dims.dx[2]/2.0)

        # Get the shear stresses (adopted from Sullivan, since same Surface parameters prescribed)
        self.u_flux = -self.friction_velocity**2 / windspeed * (PV.values[u_index,gw] + Ref.u0)
        self.v_flux = -self.friction_velocity**2 / windspeed * (PV.values[v_index,gw] + Ref.v0)

        # SurfaceBase.update(self, Gr, Ref, PV, DV, TS)
        SurfaceBase.update(self, Gr, Ref, PV, M1, M2, TS)
        return


    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
        SurfaceBase.stats_io(self, Gr, NS)
        return




cdef class SurfaceBomex(SurfaceBase):
    # def __init__(self,  LatentHeat LH):
    def __init__(self):
#         self.L_fp = LH.L_fp
#         self.Lambda_fp = LH.Lambda_fp
        self.dry_case = False
        return

    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
        SurfaceBase.initialize(self,Gr,Ref,NS)
        self.qt_flux = np.add(self.qt_flux,5.2e-5) # m/s

        self.theta_flux = 8.0e-3 # K m/s
        self.ustar_ = 0.28 #m/s
        self.theta_surface = 299.1 #K
        self.qt_surface = 22.45e-3 # kg/kg
        self.buoyancy_flux = g * ((self.theta_flux + (eps_vi-1.0)*(self.theta_surface*self.qt_flux + self.qt_surface *self.theta_flux))
                              /(self.theta_surface*(1.0 + (eps_vi-1)*self.qt_surface)))
        return

    # cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV,
    #              DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, SecondOrderMomenta M2, TimeStepping.TimeStepping TS):
#         if Pa.sub_z_rank != 0:
#             return

        cdef :
            Py_ssize_t gw = Gr.gw
            # Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t th_index = M1.name_index['th']
            Py_ssize_t qt_index = M1.name_index['qt']
            # Py_ssize_t qv_index = DV.get_varshift(Gr,'qv')

        # Get the scalar flux
        print('Surface Scheme BOMEX: wrong flux - entropy instead of pot. temperature')
        self.friction_velocity = self.ustar_
        # self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(self.theta_flux, self.qt_flux[ij], Ref.p0_half[gw],
        #                     DV.values[temp_index+ijk], PV.values[qt_index+ijk], DV.values[qv_index+ijk])


        print('!!! Surface Scheme BOMEX: wrong temperature & qv !!!')
        temperature = 293.0
        qv = 0.0
        self.friction_velocity = self.ustar_
        # self.s_flux = entropyflux_from_thetaflux_qtflux(self.theta_flux, self.qt_flux, Ref.p0_half[gw],
                        # DV.values[temp_index,gw], PV.values[qt_index,gw], DV.values[qv_index,gw])
        # self.th_flux = entropyflux_from_thetaflux_qtflux(self.theta_flux, self.qt_flux, Ref.p0_half[gw],
        #                 temperature, PV.values[qt_index,gw], qv)
        self.th_flux = 0.0

        cdef:
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            double windspeed = 0.0
            double u = M1.values[u_index,gw] + Ref.u0
            double v = M1.values[v_index,gw] + Ref.v0
        # compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0], Ref.u0, Ref.v0, self.gustiness)
        windspeed = np.fmax(np.sqrt(u*u + v*v), self.gustiness)

        # Get the shear stresses
        self.u_flux = -self.ustar_**2/windspeed * (PV.values[u_index, gw] + Ref.u0)
        self.v_flux = -self.ustar_**2/windspeed * (PV.values[v_index, gw] + Ref.v0)

        # SurfaceBase.update(self, Gr, Ref, PV, DV, TS)
        SurfaceBase.update(self, Gr, Ref, PV, M1, M2, TS)

        return


    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
        SurfaceBase.stats_io(self, Gr, NS)
        return


cdef class SurfaceGabls(SurfaceBase):
    # def __init__(self, namelist,  LatentHeat LH):
    def __init__(self, namelist):
        self.gustiness = 0.001
        self.z0 = 0.1
        # Rate of change of surface temperature, in K/hour
        # GABLS1 IC (Beare et al) value is 0.25 (given as default)
        try:
            self.cooling_rate = namelist['surface']['cooling_rate']
        except:
            self.cooling_rate = 0.25

        # self.L_fp = LH.L_fp
        # self.Lambda_fp = LH.Lambda_fp

        self.dry_case = True

        return

    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
        SurfaceBase.initialize(self,Gr,Ref,NS)
        return


#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, SecondOrderMomenta M2, TimeStepping.TimeStepping TS):
        cdef:
            Py_ssize_t gw = Gr.gw

            # Py_ssize_t s_shift = M1.get_varshift(Gr, 's')
            # Py_ssize_t t_index = DV.get_varshift(Gr, 'temperature')
            # Py_ssize_t th_index = DV.get_varshift(Gr, 'theta')
            Py_ssize_t th_index = M1.name_index['th']
            Py_ssize_t u_index = M1.name_index['u']
            Py_ssize_t v_index = M1.name_index['v']
            double u = M1.values[u_index,gw] + Ref.u0
            double v = M1.values[v_index,gw] + Ref.v0

            double windspeed = 0.0

            # double theta_rho_b, Nb2, Ri
            double zb = Gr.dz * 0.5
            # double [:] cm= np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
            double cm = 0.0
            double ch=0.0

        # compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0], Ref.u0, Ref.v0, self.gustiness)
        windspeed = np.fmax(np.sqrt(u*u + v*v), self.gustiness)

        self.T_surface = 265.0 - self.cooling_rate * TS.t/3600.0 # sst = theta_surface also

        # cdef double theta_rho_g = theta_rho_c(Ref.Pg, self.T_surface, 0.0, 0.0)
        # cdef double s_star = sd_c(Ref.Pg,self.T_surface)

        # theta_rho_b = DV.values[th_index + ijk]
        # Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
        # Ri = Nb2 * zb* zb/(windspeed[ij] * windspeed[ij])
        # exchange_coefficients_byun(Ri,zb,self.z0, &cm[ij], &ch, &self.obukhov_length[ij])
        # self.s_flux[ij] = -ch * windspeed[ij] * (PV.values[s_shift+ijk] - s_star)
        # self.friction_velocity[ij] = sqrt(cm[ij]) * windspeed[ij]
        theta_rho_b = M1.values[th_index, gw]
        # Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
        # Ri = Nb2 * zb* zb/(windspeed * windspeed)
        print('!!! Surface Scheme GABLS: compute exchange coefficienst by un !!!')
        # exchange_coefficients_byun(Ri,zb,self.z0, &cm[ij], &ch, &self.obukhov_length[ij])
        # self.s_flux[ij] = -ch * windspeed[ij] * (PV.values[s_shift+ijk] - s_star)
        self.friction_velocity = sqrt(cm) * windspeed

        # self.u_flux[ij] = -interp_2(cm[ij], cm[ij+istride_2d])*interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
        # self.v_flux[ij] = -interp_2(cm[ij], cm[ij+1])*interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)
        self.u_flux = -cm * windspeed * (PV.values[u_index,gw] + Ref.u0)
        self.v_flux = -cm * windspeed * (PV.values[v_index,gw] + Ref.v0)

        # SurfaceBase.update(self, Gr, Ref, PV, DV, TS)
        SurfaceBase.update(self, Gr, Ref, PV, M1, M2, TS)

        return


    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
        SurfaceBase.stats_io(self, Gr, NS)
        return



# Anderson, R. J., 1993: A Study of Wind Stress and Heat Flux over the Open
# Ocean by the Inertial-Dissipation Method. J. Phys. Oceanogr., 23, 2153--“2161.
# See also: ARPS documentation
cdef inline double compute_z0(double z1, double windspeed) nogil:
    cdef double z0 =z1*exp(-kappa/sqrt((0.4 + 0.079*windspeed)*1e-3))
    return z0
