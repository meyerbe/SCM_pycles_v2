from Grid cimport Grid
from ReferenceState cimport ReferenceState
from PrognosticVariables cimport PrognosticVariables
from PrognosticVariables cimport MeanVariables
# cimport DiagnosticVariables
cimport TimeStepping
# from SurfaceBudget cimport SurfaceBudget
# from Thermodynamics cimport  LatentHeat, ClausiusClapeyron
from NetCDFIO cimport NetCDFIO_Stats

cdef class SurfaceBase:
    cdef:
        double T_surface
        double th_flux
        double qt_flux
        double u_flux
        double v_flux
        double friction_velocity
        double obukhov_length
        double shf
        double lhf
        double b_flux
        bint dry_case
        double (*L_fp)(double T, double Lambda) nogil
        double (*Lambda_fp)(double T) nogil

    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS)
#     cpdef init_from_restart(self, Restart)
#     cpdef restart(self, Restart)
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, TimeStepping.TimeStepping TS)
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS)


cdef class SurfaceNone(SurfaceBase):
    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS)
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, TimeStepping.TimeStepping TS)
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS)


cdef class SurfaceSoares(SurfaceBase):
    cdef:
        double theta_flux
        # double qt_flux
        double z0
        double gustiness
        double buoyancy_flux
        double theta_surface
        double qt_surface

    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS)
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, TimeStepping.TimeStepping TS)
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS)#
    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS)


# cdef class SurfaceSoares_moist(SurfaceBase):
#     cdef:
#         double theta_flux
#         # double qt_flux
#         double z0
#         double gustiness
#         double buoyancy_flux
#         double theta_surface
#         double qt_surface
#
#     cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS)
#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS)
#     cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS)
#


cdef class SurfaceSullivanPatton(SurfaceBase):
    cdef:
        double theta_flux
        double z0
        double gustiness
        double buoyancy_flux
    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS)
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, TimeStepping.TimeStepping TS)
#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS)
#
#
cdef class SurfaceBomex(SurfaceBase):
    cdef:
        double theta_flux
        double ustar_
        double theta_surface
        double qt_surface
        double buoyancy_flux
        double gustiness


    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS)
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1,
                  TimeStepping.TimeStepping TS)
#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV,
#                   TimeStepping.TimeStepping TS)
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1, TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS)


cdef class SurfaceGabls(SurfaceBase):
    cdef:
        double gustiness
        double z0
        double cooling_rate

    cpdef initialize(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats NS)
#     cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV,
#                   TimeStepping.TimeStepping TS)
    cpdef update(self, Grid Gr, ReferenceState Ref, PrognosticVariables PV, MeanVariables M1,
                  TimeStepping.TimeStepping TS)
    cpdef stats_io(self, Grid Gr, NetCDFIO_Stats NS)


# cdef class SurfaceDYCOMS_RF01(SurfaceBase):
#     cdef:
#         double lv
#         double ft
#         double fq
#         double cm
#         double buoyancy_flux
#         double gustiness
#         double [:] windspeed
#
#     cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS)
#     cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV,
#                  TimeStepping.TimeStepping TS)
#     cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS)
#
# cdef class SurfaceDYCOMS_RF02(SurfaceBase):
#     cdef:
#         double lv
#         double ft
#         double fq
#         double ustar
#         double buoyancy_flux
#         double gustiness
#         double [:] windspeed
#
#     cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS)
#     cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS)
#     cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS)
#
# cdef class SurfaceRico(SurfaceBase):
#     cdef:
#         double cm
#         double ch
#         double cq
#         double z0
#         double gustiness
#         double s_star
#
#     cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS)
#     cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS)
#     cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS)
#
#
# cdef class SurfaceCGILS(SurfaceBase):
#     cdef:
#         Py_ssize_t loc
#         bint is_p2
#         ClausiusClapeyron CC
#         double gustiness
#         double z0
#         double ct
#
#     cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS)
#     cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS)
#     cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS)
#
#
# cdef class SurfaceZGILS(SurfaceBase):
#     cdef:
#         Py_ssize_t loc
#         bint is_p2
#         ClausiusClapeyron CC
#         double gustiness
#         double z0
#         double ct
#
#     cpdef initialize(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS)
#     cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref, PrognosticVariables.PrognosticVariables PV,
#                  DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS)
#     cpdef stats_io(self, Grid.Grid Gr, NetCDFIO_Stats NS)





cdef inline double compute_z0(double z1, double windspeed) nogil