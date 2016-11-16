#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from Grid cimport Grid
from ReferenceState cimport ReferenceState
from PrognosticVariables cimport PrognosticVariables
from PrognosticVariables cimport MeanVariables
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
        casename = namelist['meta']['casename']
        # if casename == 'SullivanPatton':
        #    return SurfaceSullivanPatton(LH)
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

#     cpdef init_from_restart(self, Restart):
#         self.T_surface = Restart.restart_data['surf']['T_surf']
#         return
#     cpdef restart(self, Restart):
#         Restart.restart_data['surf'] = {}
#         Restart.restart_data['surf']['T_surf'] = self.T_surface
#         return
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, TimeStepping.TimeStepping TS):
        # DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
        cdef :
            Py_ssize_t gw = Gr.dims.gw
#             Py_ssize_t t_index = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t th_index = M1.get_varshift(Gr, 's')
            Py_ssize_t u_index = M1.get_varshift(Gr, 'u')
            Py_ssize_t v_index = M1.get_varshift(Gr, 'v')
            Py_ssize_t ql_index, qt_index
            double [:] t_mean = np.zeros(Gr.nzg) #=  Pa.HorizontalMean(Gr, &DV.values[t_shift])
#             double cp_, lam, lv, pv, pd, sv, sd
            double dzi = 1.0/Gr.dims.dz
            double tendency_factor = Ref.alpha0_half[gw]/Ref.alpha0[gw-1]*dzi
#
        if self.dry_case:
            # with nogil:
            temperature = 293.0
            t_mean[gw] = temperature
            print('!!! not correct temperature in Surface Base update !!!')
            self.shf = self.s_flux * Ref.rho0_half[gw] * temperature#DV.values[t_index,gw]
            self.b_flux = self.shf * g * Ref.alpha0_half[gw]/cpd/t_mean[gw]
            self.obukhov_length = -self.friction_velocity * self.friction_velocity * self.friction_velocity /self.b_flux/vkb

            PV.tendencies[u_index,gw] +=  self.u_flux * tendency_factor
            PV.tendencies[v_index,gw] +=  self.v_flux * tendency_factor
            PV.tendencies[th_index,gw] +=  self.th_flux * tendency_factor

        else:
            print('Surface Base: get DV ql')
            # ql_index = DV.get_varshift(Gr,'ql')
            qt_index = M1.get_varshift(Gr, 'qt')
        #     with nogil:
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

            PV.tendencies[u_index,gw] +=  self.u_flux * tendency_factor
            PV.tendencies[v_index,gw] +=  self.v_flux * tendency_factor
            PV.tendencies[th_index,gw] +=  self.th_flux * tendency_factor
            PV.tendencies[qt_index,gw] +=  self.qt_flux * tendency_factor

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
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, TimeStepping.TimeStepping TS):
    #                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
        return
    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
        return


# # _____________________
# # SOARES
# # like in Sullivan case: z0 is given (ustar_fixed = 'False')
# # like in Bomex case: surface heat and moisture flux constant and prescribed
cdef class SurfaceSoares(SurfaceBase):
    def __init__(self):
        self.z0 = 0.001 #m (Roughness length)
        self.gustiness = 0.001 #m/s, minimum surface windspeed for determination of u*
        return

#     @cython.boundscheck(False)  #Turn off numpy array index bounds checking
#     @cython.wraparound(False)   #Turn off numpy array wrap around indexing
#     @cython.cdivision(True)
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
        T0 = Ref.p0_half[Gr.dims.gw] * Ref.alpha0_half[Gr.dims.gw]/Rd
        self.buoyancy_flux = self.theta_flux * exner(Ref.p0_half[Gr.dims.gw]) * g /T0

        return

#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     @cython.cdivision(True)
# # update adopted and modified from Sullivan + Bomex
#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, TimeStepping.TimeStepping TS):
#         # Since this case is completely dry, the computation of entropy flux from sensible heat flux is very simple
        cdef:
#             Py_ssize_t i, j, ijk, ij
            Py_ssize_t gw = Gr.dims.gw
#             Py_ssize_t imax = Gr.dims.nlg[0]
#             Py_ssize_t jmax = Gr.dims.nlg[1]
#             Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
#             Py_ssize_t jstride = Gr.dims.nlg[2]
#             Py_ssize_t istride_2d = Gr.dims.nlg[1]
#             Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t th_index = PV.get_varshift(Gr, 's')
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
            Py_ssize_t u_index = M1.get_varshift(Gr, 'u')
            Py_ssize_t v_index = M1.get_varshift(Gr, 'v')
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
        SurfaceBase.update(self, Gr, Ref, PV, M1, TS)
        return


    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
        SurfaceBase.stats_io(self, Gr, NS)
        return



# cdef class SurfaceSoares_moist(SurfaceBase):
#     def __init__(self, LatentHeat LH):
#         self.z0 = 0.001 #m (Roughness length)
#         self.gustiness = 0.001 #m/s, minimum surface windspeed for determination of u*
#
#         self.L_fp = LH.L_fp
#         self.Lambda_fp = LH.Lambda_fp
#         self.dry_case = False
#         return
#
#     @cython.boundscheck(False)  #Turn off numpy array index bounds checking
#     @cython.wraparound(False)   #Turn off numpy array wrap around indexing
#     @cython.cdivision(True)
#     cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):     # Sullivan
#         SurfaceBase.initialize(self,Gr,Ref,NS)
#
#         # ### Bomex
#         # self.qt_flux = np.add(self.qt_flux,5.2e-5) # m/s
#         # self.theta_flux = 8.0e-3 # K m/s
#         # # self.ustar_ = 0.28 #m/s
#         # self.theta_surface = 299.1 #K
#         # self.qt_surface = 22.45e-3 # kg/kg
#
#         ### Soares_moist
#         # self.qt_flux = 5.2e-5 # m/s (Soares: 2.5e-5) (Bomex: 5.2e-5)
#         self.qt_flux = np.add(self.qt_flux,2.5e-5)
#         # self.qt_flux = np.add(self.qt_flux,0.0)
#         self.theta_flux = 8.0e-3 # K m/s (Bomex)
#         self.theta_surface = 300.0 # K
#         self.qt_surface = 5.0e-3 # kg/kg
#         #
#         # # Bomex:
#         self.buoyancy_flux = g * ((self.theta_flux + (eps_vi-1.0)*(self.theta_surface*self.qt_flux[0]
#                                                                    + self.qt_surface *self.theta_flux))
#                               /(self.theta_surface*(1.0 + (eps_vi-1)*self.qt_surface)))
#         # # Sullivan:
#         # # T0 = Ref.p0_half[Gr.dims.gw] * Ref.alpha0_half[Gr.dims.gw]/Rd
#         # # self.buoyancy_flux = self.theta_flux * exner(Ref.p0_half[Gr.dims.gw]) * g /T0
#
#         return
#
#     @cython.boundscheck(False)  #Turn off numpy array index bounds checking
#     @cython.wraparound(False)   #Turn off numpy array wrap around indexing
#     @cython.cdivision(True)
# # # update adopted and modified from Sullivan + Bomex
#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
#         # Since this case is completely dry, the computation of entropy flux from sensible heat flux is very simple
#         cdef:
#             Py_ssize_t i, j, ij, ijk
#             Py_ssize_t gw = Gr.dims.gw
#             Py_ssize_t imax = Gr.dims.nlg[0]
#             Py_ssize_t jmax = Gr.dims.nlg[1]
#             Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
#             Py_ssize_t jstride = Gr.dims.nlg[2]
#             Py_ssize_t istride_2d = Gr.dims.nlg[1]
#             Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
#             Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
#             Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
#             Py_ssize_t qv_shift = DV.get_varshift(Gr,'qv')
#
#         # Scalar fluxes (adopted from Bomex)
#         with nogil:
#             for i in xrange(imax):
#                 for j in xrange(jmax):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     # Sullivan
#                     # self.s_flux[ij] = cpd * self.theta_flux*exner(Ref.p0_half[gw])/DV.values[temp_shift+ijk]
#                     # Bomex (entropy flux includes qt flux)
#                     self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(self.theta_flux, self.qt_flux[ij], Ref.p0_half[gw],
#                                                                         DV.values[temp_shift+ijk], PV.values[qt_shift+ijk], DV.values[qv_shift+ijk])
#
#         # Windspeed (adopted from Sullivan, equivalent to Bomex)
#         cdef:
#             Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
#             Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
#             double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1],dtype=np.double,order='c')
#         compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)
#
#        # Surface Values: friction velocity, obukhov lenght (adopted from Sullivan, since same Surface parameters prescribed)
#        #  cdef :
#             # Py_ssize_t lmo_shift = DV.get_varshift_2d(Gr, 'obukhov_length')
#             # Py_ssize_t ustar_shift = DV.get_varshift_2d(Gr, 'friction_velocity')
#         with nogil:
#             for i in xrange(1,imax):
#                 for j in xrange(1,jmax):
#                     ij = i * istride_2d + j
#                     self.friction_velocity[ij] = compute_ustar(windspeed[ij],self.buoyancy_flux,self.z0, Gr.dims.dx[2]/2.0)
#                     # self.obukhov_length[ij] = -self.friction_velocity[ij]*self.friction_velocity[ij]*self.friction_velocity[ij]/self.buoyancy_flux/vkb
#
#         # Get the shear stresses (adopted from Sullivan, since same Surface parameters prescribed)
#             for i in xrange(1,imax-1):
#                 for j in xrange(1,jmax-1):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     self.u_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+istride_2d])**2/interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
#                     self.v_flux[ij] = -interp_2(self.friction_velocity[ij], self.friction_velocity[ij+1])**2/interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)
#                     # PV.tendencies[u_shift + ijk] += self.u_flux[ij] * tendency_factor
#                     # PV.tendencies[v_shift + ijk] += self.v_flux[ij] * tendency_factor
#
#         SurfaceBase.update(self, Gr, Ref, PV, DV, TS)
#         return
#
#     cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS):
#         SurfaceBase.stats_io(self, Gr, NS)
#         return
#
#
#

# SULLIVAN
cdef class SurfaceSullivanPatton(SurfaceBase):
    def __init__(self):
        self.theta_flux = 0.24 # K m/s
        self.z0 = 0.1 #m (Roughness length)
        self.gustiness = 0.001 #m/s, minimum surface windspeed for determination of u*

        self.dry_case = True
        return

    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):

        T0 = Ref.p0_half[Gr.dims.gw] * Ref.alpha0_half[Gr.dims.gw]/Rd
        self.buoyancy_flux = self.theta_flux * exner(Ref.p0_half[Gr.dims.gw]) * g /T0
        SurfaceBase.initialize(self,Gr,Ref,NS)

        return


#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, TimeStepping.TimeStepping TS):
#         # Since this case is completely dry, the computation of entropy flux from sensible heat flux is very simple
        cdef:
            Py_ssize_t gw = Gr.dims.gw

        print('!!! Surface Scheme Sullivan: wrong temperature !!!')
#             Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')

        #Get the scalar flux (dry entropy only)
#         with nogil:
#             for i in xrange(imax):
#                 for j in xrange(jmax):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     self.s_flux[ij] = cpd * self.theta_flux*exner(Ref.p0_half[gw])/DV.values[temp_shift+ijk]
        temperature = 293.0
        # self.s_flux = cpd * self.theta_flux*exner(Ref.p0_half[gw])/DV.values[temp_shift,gw]
        self.s_flux = cpd * self.theta_flux*exner(Ref.p0_half[gw])/temperature

        cdef:
            Py_ssize_t u_index = M1.get_varshift(Gr, 'u')
            Py_ssize_t v_index = M1.get_varshift(Gr, 'v')
            double windspeed = 0.0
            double u = M1.values[u_index,gw] + Ref.u0
            double v = M1.values[v_index,gw] + Ref.v0
#         compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed,Ref.u0, Ref.v0,self.gustiness)
        windspeed = np.fmax(np.sqrt(u*u + v*v), self.gustiness)

        # Get the shear stresses
        print('!!! Surface Scheme Sullivan: compute ustar / friction velocity !!!')
        # self.friction_velocity = compute_ustar(windspeed,self.buoyancy_flux,self.z0, Gr.dims.dz/2.0)
        self.friction_velocity = 1.0
        self.u_flux = -self.friction_velocity**2 / windspeed * (PV.values[u_index,gw] + Ref.u0)
        self.v_flux = -self.friction_velocity**2 / windspeed * (PV.values[v_index,gw] + Ref.v0)

        # SurfaceBase.update(self, Gr, Ref, PV, DV, TS)
        SurfaceBase.update(self, Gr, Ref, PV, M1, TS)
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
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, TimeStepping.TimeStepping TS):
#         if Pa.sub_z_rank != 0:
#             return

        cdef :
            Py_ssize_t gw = Gr.dims.gw
            # Py_ssize_t temp_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t th_index = M1.get_varshift(Gr, 's')
            Py_ssize_t qt_index = M1.get_varshift(Gr, 'qt')
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
            Py_ssize_t u_index = M1.get_varshift(Gr, 'u')
            Py_ssize_t v_index = M1.get_varshift(Gr, 'v')
            double windspeed = 0.0
            double u = M1.values[u_index,gw] + Ref.u0
            double v = M1.values[v_index,gw] + Ref.v0
        # compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0], Ref.u0, Ref.v0, self.gustiness)
        windspeed = np.fmax(np.sqrt(u*u + v*v), self.gustiness)

        # Get the shear stresses
        self.u_flux = -self.ustar_**2/windspeed * (PV.values[u_index, gw] + Ref.u0)
        self.v_flux = -self.ustar_**2/windspeed * (PV.values[v_index, gw] + Ref.v0)

        # SurfaceBase.update(self, Gr, Ref, PV, DV, TS)
        SurfaceBase.update(self, Gr, Ref, PV, M1, TS)

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
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, TimeStepping.TimeStepping TS):
        cdef:
            Py_ssize_t gw = Gr.dims.gw

            # Py_ssize_t s_shift = M1.get_varshift(Gr, 's')
            # Py_ssize_t t_index = DV.get_varshift(Gr, 'temperature')
            # Py_ssize_t th_index = DV.get_varshift(Gr, 'theta')
            Py_ssize_t th_index = M1.get_varshift(Gr, 'theta')
            Py_ssize_t u_index = M1.get_varshift(Gr, 'u')
            Py_ssize_t v_index = M1.get_varshift(Gr, 'v')
            double u = M1.values[u_index,gw] + Ref.u0
            double v = M1.values[v_index,gw] + Ref.v0

            double windspeed = 0.0

            # double theta_rho_b, Nb2, Ri
            double zb = Gr.dims.dz * 0.5
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
        SurfaceBase.update(self, Gr, Ref, PV, M1, TS)

        return


    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
        SurfaceBase.stats_io(self, Gr, NS)
        return


# cdef class SurfaceDYCOMS_RF01(SurfaceBase):
#     def __init__(self,namelist, LatentHeat LH):
#         self.ft = 15.0
#         self.fq = 115.0
#         self.gustiness = 0.0
#         self.cm = 0.0011
#         self.L_fp = LH.L_fp
#         self.Lambda_fp = LH.Lambda_fp
#         sst = 292.5 # K
#         psurface = 1017.8e2 # Pa
#         theta_surface = sst/exner(psurface)
#         qt_surface = 13.84e-3 # qs(sst) using Teten's formula
#         density_surface = 1.22 #kg/m^3
#         theta_flux = self.ft/(density_surface*cpm(qt_surface)*exner(psurface))
#         qt_flux_ = self.fq/self.L_fp(sst,self.Lambda_fp(sst))
#         self.buoyancy_flux = g * ((theta_flux + (eps_vi-1.0)*(theta_surface*qt_flux_ + qt_surface * theta_flux))
#                               /(theta_surface*(1.0 + (eps_vi-1)*qt_surface)))
#
#         self.dry_case = False
#
#     cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
#         SurfaceBase.initialize(self,Gr,Ref,NS)
#         self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
#         self.T_surface = 292.5
#
#         return
#
#
#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
#
#         if Pa.sub_z_rank != 0:
#             return
#
#         cdef:
#             Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
#             Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
#             Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
#             Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
#             Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
#             Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
#
#         compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &self.windspeed[0],Ref.u0, Ref.v0,self.gustiness)
#
#         cdef:
#             Py_ssize_t i,j, ijk, ij
#             Py_ssize_t gw = Gr.dims.gw
#             Py_ssize_t imax = Gr.dims.nlg[0]
#             Py_ssize_t jmax = Gr.dims.nlg[1]
#             Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
#             Py_ssize_t jstride = Gr.dims.nlg[2]
#             Py_ssize_t istride_2d = Gr.dims.nlg[1]
#
#             double lam, lv, pv, pd, sv, sd
#
#             double [:] windspeed = self.windspeed
#
#
#         with nogil:
#             for i in xrange(gw-1, imax-gw+1):
#                 for j in xrange(gw-1, jmax-gw+1):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     self.friction_velocity[ij] = sqrt(self.cm) * self.windspeed[ij]
#                     lam = self.Lambda_fp(DV.values[t_shift+ijk])
#                     lv = self.L_fp(DV.values[t_shift+ijk],lam)
#                     pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
#                     pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
#                     sv = sv_c(pv,DV.values[t_shift+ijk])
#                     sd = sd_c(pd,DV.values[t_shift+ijk])
#                     self.qt_flux[ij] = self.fq / lv / 1.22
#                     self.s_flux[ij] = Ref.alpha0_half[gw] * (self.ft/DV.values[t_shift+ijk] + self.fq*(sv - sd)/lv)
#             for i in xrange(gw, imax-gw):
#                 for j in xrange(gw, jmax-gw):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     self.u_flux[ij] = -self.cm * interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
#                     self.v_flux[ij] = -self.cm * interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)
#
#         SurfaceBase.update(self, Gr, Ref, PV, DV,TS)
#         return
#
#     cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
#         SurfaceBase.stats_io(self, Gr, NS)
#
#         return
#
#
#
#
# cdef class SurfaceDYCOMS_RF02(SurfaceBase):
#     def __init__(self,namelist, LatentHeat LH):
#         self.ft = 16.0
#         self.fq = 93.0
#         self.gustiness = 0.0
#         self.ustar = 0.25
#         self.L_fp = LH.L_fp
#         self.Lambda_fp = LH.Lambda_fp
#         sst = 292.5 # K
#         psurface = 1017.8e2 # Pa
#         theta_surface = sst/exner(psurface)
#         qt_surface = 13.84e-3 # qs(sst) using Teten's formula
#         density_surface = 1.22 #kg/m^3
#         theta_flux = self.ft/(density_surface*cpm(qt_surface)*exner(psurface))
#         qt_flux_ = self.fq/self.L_fp(sst,self.Lambda_fp(sst))
#         self.buoyancy_flux = g * ((theta_flux + (eps_vi-1.0)*(theta_surface*qt_flux_ + qt_surface * theta_flux))
#                               /(theta_surface*(1.0 + (eps_vi-1)*qt_surface)))
#
#         self.dry_case = False
#
#     cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
#         SurfaceBase.initialize(self,Gr,Ref,NS)
#         self.windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
#         self.T_surface = 292.5 # assuming same sst as DYCOMS RF01
#
#         return
#
#
#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
#
#         if Pa.sub_z_rank != 0:
#             return
#
#         cdef:
#             Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
#             Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
#             Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
#             Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
#             Py_ssize_t ql_shift = DV.get_varshift(Gr, 'ql')
#
#
#
#         compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &self.windspeed[0],Ref.u0, Ref.v0,self.gustiness)
#
#         cdef:
#             Py_ssize_t i,j, ijk, ij
#             Py_ssize_t gw = Gr.dims.gw
#             Py_ssize_t imax = Gr.dims.nlg[0]
#             Py_ssize_t jmax = Gr.dims.nlg[1]
#             Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
#             Py_ssize_t jstride = Gr.dims.nlg[2]
#             Py_ssize_t istride_2d = Gr.dims.nlg[1]
#
#             double tendency_factor = Ref.alpha0_half[gw]/Ref.alpha0[gw-1]/Gr.dims.dx[2]
#             double lam
#             double lv
#             double pv
#             double pd
#             double sv
#             double sd
#
#             double [:] windspeed = self.windspeed
#
#         with nogil:
#             for i in xrange(gw-1, imax-gw+1):
#                 for j in xrange(gw-1, jmax-gw+1):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     self.friction_velocity[ij] = self.ustar
#                     lam = self.Lambda_fp(DV.values[t_shift+ijk])
#                     lv = self.L_fp(DV.values[t_shift+ijk],lam)
#                     pv = pv_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
#                     pd = pd_c(Ref.p0_half[gw], PV.values[ijk + qt_shift], PV.values[ijk + qt_shift] - DV.values[ijk + ql_shift])
#                     sv = sv_c(pv,DV.values[t_shift+ijk])
#                     sd = sd_c(pd,DV.values[t_shift+ijk])
#                     self.qt_flux[ij] = self.fq / lv / 1.21
#                     self.s_flux[ij] = Ref.alpha0_half[gw] * (self.ft/DV.values[t_shift+ijk] + self.fq*(sv - sd)/lv)
#             for i in xrange(gw, imax-gw):
#                 for j in xrange(gw, jmax-gw):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     self.u_flux[ij] = -self.ustar*self.ustar / interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
#                     self.v_flux[ij] = -self.ustar*self.ustar / interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)
#
#         SurfaceBase.update(self, Gr, Ref, PV, DV, TS)
#
#         return
#
#     cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
#         SurfaceBase.stats_io(self, Gr, NS)
#
#         return
#
#
#
#
# cdef class SurfaceRico(SurfaceBase):
#     def __init__(self, LatentHeat LH):
#         self.cm =0.001229
#         self.ch = 0.001094
#         self.cq = 0.001133
#         self.z0 = 0.00015
#         self.gustiness = 0.0
#         self.L_fp = LH.L_fp
#         self.Lambda_fp = LH.Lambda_fp
#         self.dry_case = False
#         return
#
#
#     cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
#         SurfaceBase.initialize(self,Gr,Ref,NS)
#
#         self.cm = self.cm*(log(20.0/self.z0)/log(Gr.zl_half[Gr.dims.gw]/self.z0))**2
#         self.ch = self.ch*(log(20.0/self.z0)/log(Gr.zl_half[Gr.dims.gw]/self.z0))**2
#         self.cq = self.cq*(log(20.0/self.z0)/log(Gr.zl_half[Gr.dims.gw]/self.z0))**2
#
#
#         cdef double pv_star = pv_c(Ref.Pg, Ref.qtg, Ref.qtg)
#         cdef double  pd_star = Ref.Pg - pv_star
#         self.s_star = (1.0-Ref.qtg) * sd_c(pd_star, Ref.Tg) + Ref.qtg * sv_c(pv_star,Ref.Tg)
#
#         return
#
#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
#
#         if Pa.sub_z_rank != 0:
#             return
#
#         cdef:
#             Py_ssize_t i,j, ijk, ij
#             Py_ssize_t gw = Gr.dims.gw
#             Py_ssize_t imax = Gr.dims.nlg[0]
#             Py_ssize_t jmax = Gr.dims.nlg[1]
#             Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
#             Py_ssize_t jstride = Gr.dims.nlg[2]
#             Py_ssize_t istride_2d = Gr.dims.nlg[1]
#             Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
#             Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
#             Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
#             Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
#             Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
#
#             double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
#             double ustar_
#             double buoyancy_flux, theta_flux
#             double theta_surface = Ref.Tg * exner(Ref.Pg)
#
#             double cm_sqrt = sqrt(self.cm)
#
#         compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0],Ref.u0, Ref.v0,self.gustiness)
#
#         with nogil:
#             for i in xrange(gw, imax-gw):
#                 for j in xrange(gw,jmax-gw):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     theta_flux = -self.ch * windspeed[ij] * (DV.values[t_shift + ijk]*exner(Ref.p0_half[gw]) - theta_surface)
#
#                     self.s_flux[ij]  = -self.ch * windspeed[ij] * (PV.values[s_shift + ijk] - self.s_star)
#                     self.qt_flux[ij] = -self.cq * windspeed[ij] * (PV.values[qt_shift + ijk] - Ref.qtg)
#                     buoyancy_flux = g * ((theta_flux + (eps_vi-1.0)*(theta_surface*self.qt_flux[ij] + Ref.qtg * theta_flux))/(theta_surface*(1.0 + (eps_vi-1)*Ref.qtg)))
#                     self.u_flux[ij]  = -self.cm * interp_2(windspeed[ij], windspeed[ij + istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
#                     self.v_flux[ij] = -self.cm * interp_2(windspeed[ij], windspeed[ij + 1])* (PV.values[v_shift + ijk] + Ref.v0)
#                     ustar_ = cm_sqrt * windspeed[ij]
#                     self.friction_velocity[ij] = ustar_
#
#         SurfaceBase.update(self, Gr, Ref, PV, DV, TS)
#
#         return
#
#
#
#
#     cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
#         SurfaceBase.stats_io(self, Gr, NS)
#         return
#
#
#
#
#
#
# cdef class SurfaceCGILS(SurfaceBase):
#     def __init__(self, namelist, LatentHeat LH):
#         try:
#             self.loc = namelist['meta']['CGILS']['location']
#             if self.loc !=12 and self.loc != 11 and self.loc != 6:
#                 Pa.root_print('Invalid CGILS location (must be 6, 11, or 12)')
#                 Pa.kill()
#         except:
#             Pa.root_print('Must provide a CGILS location (6/11/12) in namelist')
#             Pa.kill()
#         try:
#             self.is_p2 = namelist['meta']['CGILS']['P2']
#         except:
#             Pa.root_print('Must specify if CGILS run is perturbed')
#             Pa.kill()
#
#         self.gustiness = 0.001
#         self.z0 = 1.0e-4
#         self.L_fp = LH.L_fp
#         self.Lambda_fp = LH.Lambda_fp
#         self.CC = ClausiusClapeyron()
#         self.CC.initialize(namelist, LH)
#         self.dry_case = False
#
#         return
#
#     cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
#         SurfaceBase.initialize(self,Gr,Ref,NS)
#
#         # Find the scalar transfer coefficient consistent with the vertical grid spacing
#         cdef double z1 = Gr.dims.dx[2] * 0.5
#         cdef double cq = 1.2e-3
#         cdef double u10m=0.0, ct_ic=0.0, z1_ic=0.0
#         if self.loc == 12:
#             ct_ic = 0.0104
#             z1_ic = 2.5
#         elif self.loc == 11:
#             ct_ic = 0.0081
#             z1_ic = 12.5
#         elif self.loc == 6:
#             ct_ic = 0.0081
#             z1_ic = 20.0
#
#         u10m = ct_ic/cq * np.log(z1_ic/self.z0)**2/np.log(10.0/self.z0)**2
#
#         self.ct = cq * u10m * (np.log(10.0/self.z0)/np.log(z1/self.z0))**2
#
#
#         return
#
#
#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
#
#         if Pa.sub_z_rank != 0:
#             return
#
#         cdef:
#             Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
#             Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
#             Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
#             Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
#             Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
#             double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
#
#         compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0], Ref.u0, Ref.v0, self.gustiness)
#
#         cdef:
#             Py_ssize_t i,j, ijk, ij
#             Py_ssize_t gw = Gr.dims.gw
#             Py_ssize_t imax = Gr.dims.nlg[0]
#             Py_ssize_t jmax = Gr.dims.nlg[1]
#             Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
#             Py_ssize_t jstride = Gr.dims.nlg[2]
#             Py_ssize_t istride_2d = Gr.dims.nlg[1]
#             double zb = Gr.dims.dx[2] * 0.5
#             double [:] cm = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
#             double pv_star = self.CC.LT.fast_lookup(self.T_surface)
#             double qv_star = eps_v * pv_star/(Ref.Pg + (eps_v-1.0)*pv_star)
#             double [:] t_mean = Pa.HorizontalMean(Gr, &DV.values[t_shift])
#             double buoyancy_flux, th_flux
#             double exner_b = exner(Ref.p0_half[gw])
#             double theta_0 = self.T_surface/exner(Ref.Pg)
#
#
#
#         with nogil:
#             for i in xrange(gw-1, imax-gw+1):
#                 for j in xrange(gw-1,jmax-gw+1):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     self.qt_flux[ij] = self.ct * (0.98 * qv_star - PV.values[qt_shift + ijk])
#                     th_flux = self.ct * (theta_0 - DV.values[t_shift + ijk]/exner_b )
#                     buoyancy_flux = g * th_flux * exner_b/t_mean[gw] + g * (eps_vi-1.0)*self.qt_flux[ij]
#
#                     self.friction_velocity[ij] = compute_ustar(windspeed[ij],buoyancy_flux,self.z0, zb)
#                     self.s_flux[ij] = entropyflux_from_thetaflux_qtflux(th_flux, self.qt_flux[ij],
#                                                                         Ref.p0_half[gw], DV.values[t_shift + ijk],
#                                                                         PV.values[qt_shift + ijk], PV.values[qt_shift + ijk])
#                     cm[ij] = (self.friction_velocity[ij]/windspeed[ij]) *  (self.friction_velocity[ij]/windspeed[ij])
#
#
#             for i in xrange(gw, imax-gw):
#                 for j in xrange(gw, jmax-gw):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     self.u_flux[ij] = -interp_2(cm[ij], cm[ij+istride_2d])*interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
#                     self.v_flux[ij] = -interp_2(cm[ij], cm[ij+1])*interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)
#
#         SurfaceBase.update(self, Gr, Ref, PV, DV, TS)
#
#         return
#
#
#     cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
#         SurfaceBase.stats_io(self, Gr, NS)
#
#         return
#
#
#
# cdef class SurfaceZGILS(SurfaceBase):
#     def __init__(self, namelist, LatentHeat LH):
#
#
#         self.gustiness = 0.001
#         self.z0 = 1.0e-3
#         self.L_fp = LH.L_fp
#         self.Lambda_fp = LH.Lambda_fp
#         self.CC = ClausiusClapeyron()
#         self.CC.initialize(namelist, LH)
#
#         self.dry_case = False
#         try:
#             self.loc = namelist['meta']['ZGILS']['location']
#             if self.loc !=12 and self.loc != 11 and self.loc != 6:
#                 Pa.root_print('SURFACE: Invalid ZGILS location (must be 6, 11, or 12) '+ str(self.loc))
#                 Pa.kill()
#         except:
#             Pa.root_print('SURFACE: Must provide a ZGILS location (6/11/12) in namelist')
#             Pa.kill()
#
#
#         return
#
#     cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS):
#         SurfaceBase.initialize(self,Gr,Ref,NS)
#         # Set the initial sst value to the Fixed-SST case value (Tan et al 2016a, Table 1)
#         if self.loc == 12:
#             self.T_surface  = 289.75
#         elif self.loc == 11:
#             self.T_surface = 292.22
#         elif self.loc == 6:
#             self.T_surface = 298.86
#         return
#
#
#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS):
#
#         if Pa.sub_z_rank != 0:
#             return
#
#         cdef:
#             Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
#             Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
#             Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
#             Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
#             Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
#             Py_ssize_t th_shift = DV.get_varshift(Gr, 'theta_rho')
#             double [:] windspeed = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
#
#         compute_windspeed(&Gr.dims, &PV.values[u_shift], &PV.values[v_shift], &windspeed[0], Ref.u0, Ref.v0, self.gustiness)
#
#         cdef:
#             Py_ssize_t i,j, ijk, ij
#             Py_ssize_t gw = Gr.dims.gw
#             Py_ssize_t imax = Gr.dims.nlg[0]
#             Py_ssize_t jmax = Gr.dims.nlg[1]
#             Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
#             Py_ssize_t jstride = Gr.dims.nlg[2]
#             Py_ssize_t istride_2d = Gr.dims.nlg[1]
#
#
#             double ustar, t_flux, b_flux
#             double theta_rho_b, Nb2, Ri
#             double zb = Gr.dims.dx[2] * 0.5
#             double [:] cm = np.zeros(Gr.dims.nlg[0]*Gr.dims.nlg[1], dtype=np.double, order='c')
#             double ch=0.0
#
#             double pv_star = self.CC.LT.fast_lookup(self.T_surface)
#             double qv_star = eps_v * pv_star/(Ref.Pg + (eps_v-1.0)*pv_star)
#
#
#
#             # Find the surface entropy
#             double pd_star = Ref.Pg - pv_star
#
#             double theta_rho_g = theta_rho_c(Ref.Pg, self.T_surface, qv_star, qv_star)
#             double s_star = sd_c(pd_star,self.T_surface) * (1.0 - qv_star) + sv_c(pv_star, self.T_surface) * qv_star
#
#             double [:] t_mean = Pa.HorizontalMean(Gr, &DV.values[t_shift])
#
#         with nogil:
#             for i in xrange(gw-1, imax-gw+1):
#                 for j in xrange(gw-1,jmax-gw+1):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     theta_rho_b = DV.values[th_shift + ijk]
#                     Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
#                     Ri = Nb2 * zb * zb/(windspeed[ij] * windspeed[ij])
#                     exchange_coefficients_byun(Ri, zb, self.z0, &cm[ij], &ch, &self.obukhov_length[ij])
#                     self.s_flux[ij] = -ch *windspeed[ij] * (PV.values[s_shift + ijk] - s_star)
#                     self.qt_flux[ij] = -ch *windspeed[ij] *  (PV.values[qt_shift + ijk] - qv_star)
#                     ustar = sqrt(cm[ij]) * windspeed[ij]
#                     self.friction_velocity[ij] = ustar
#
#             for i in xrange(gw, imax-gw):
#                 for j in xrange(gw, jmax-gw):
#                     ijk = i * istride + j * jstride + gw
#                     ij = i * istride_2d + j
#                     self.u_flux[ij] = -interp_2(cm[ij], cm[ij+istride_2d])*interp_2(windspeed[ij], windspeed[ij+istride_2d]) * (PV.values[u_shift + ijk] + Ref.u0)
#                     self.v_flux[ij] = -interp_2(cm[ij], cm[ij+1])*interp_2(windspeed[ij], windspeed[ij+1]) * (PV.values[v_shift + ijk] + Ref.v0)
#
#         SurfaceBase.update(self, Gr, Ref, PV, DV, TS)
#
#         return
#
#
#     cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS):
#         SurfaceBase.stats_io(self, Gr, NS)
#
#         return
#
#

# Anderson, R. J., 1993: A Study of Wind Stress and Heat Flux over the Open
# Ocean by the Inertial-Dissipation Method. J. Phys. Oceanogr., 23, 2153--“2161.
# See also: ARPS documentation
cdef inline double compute_z0(double z1, double windspeed) nogil:
    cdef double z0 =z1*exp(-kappa/sqrt((0.4 + 0.079*windspeed)*1e-3))
    return z0
