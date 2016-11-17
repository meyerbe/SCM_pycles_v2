


import numpy as np
cimport numpy as np
# math modules act only on scalars, numpy modules can be applied on whole arrays and matrices
from math import atan, sqrt, log

from Grid cimport Grid
include 'parameters.pxi'

from thermodynamic_functions cimport pd_c, pv_c, exner, cpm_c
from thermodynamic_functions cimport theta_rho


'''Stability Functions'''
cpdef inline double psi_m_unstable(double zeta, double zeta0):
    cdef:
        # const double x = np.pow((1.0 - gamma_m * zeta),0.25)
        # const double x0 = pow((1.0 - gamma_m * zeta0), 0.25)
        # double psi_m = 2.0 * log((1.0 + x)/(1.0 + x0)) + log((1.0 + x*x)/(1.0 + x0 * x0))-2.0*np.arctan(x)+2.0*np.arctan(x0)
        double x = (1.0 - gamma_m * zeta)**0.25
        double x0 = (1.0 - gamma_m * zeta0)**0.25
        double psi_m = 2.0 * log((1.0 + x)/(1.0 + x0)) + log((1.0 + x*x)/(1.0 + x0 * x0))-2.0*atan(x)+2.0*atan(x0)
    return psi_m

cpdef inline double psi_h_unstable(double zeta, double zeta0):
    # cdef const double y, y0, psi_h
        # const double y = np.sqrt(1.0 - gamma_h * zeta )
        # const double y0 = np.sqrt(1.0 - gamma_h * zeta0 )
        # double psi_h = 2.0 * np.log((1.0 + y)/(1.0+y0))
    cdef:
        double y = sqrt(1.0 - gamma_h * zeta )
        double y0 = sqrt(1.0 - gamma_h * zeta0 )
        double psi_h = 2.0 * log((1.0 + y)/(1.0+y0))
    return psi_h

cpdef inline double psi_m_stable(double zeta, double zeta0):
    cdef:
        double psi_m = -beta_m * (zeta - zeta0)
    return psi_m

cpdef inline double psi_h_stable(double zeta, double zeta0):
    cdef double psi_h = -beta_h * (zeta - zeta0)
    return psi_h

'''Friction Velocity'''
cpdef double compute_ustar(double windspeed, double buoyancy_flux, double z0, double zb):
    cdef:
        double lmo, zeta, zeta0, psi_m, ustar
        double ustar0, ustar1, ustar_new, f0, f1, delta_ustar
        double logz = log(zb/z0)

    #//use neutral condition as first guess
    ustar0 = windspeed * vkb/logz
    if(np.abs(buoyancy_flux) > 1.0e-20):
        lmo = -ustar0 * ustar0 * ustar0/(buoyancy_flux * vkb)
        zeta = zb/lmo
        zeta0 = z0/lmo
        if(zeta >= 0.0):
            f0 = windspeed - ustar0/vkb*(logz - psi_m_stable(zeta,zeta0))
            ustar1 = windspeed*vkb/(logz - psi_m_stable(zeta,zeta0))
            lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * vkb)
            zeta = zb/lmo
            zeta0 = z0/lmo
            f1 = windspeed - ustar1/vkb*(logz - psi_m_stable(zeta,zeta0))
            ustar = ustar1
            delta_ustar = ustar1 -ustar0
            while(np.abs(delta_ustar) > 1e-10):
                ustar_new = ustar1 - f1 * delta_ustar/(f1-f0)
                f0 = f1
                ustar0 = ustar1
                ustar1 = ustar_new
                lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * vkb)
                zeta = zb/lmo
                zeta0 = z0/lmo
                f1 = windspeed - ustar1/vkb*(logz - psi_m_stable(zeta,zeta0))
                delta_ustar = ustar1 -ustar0
        else:
            f0 = windspeed - ustar0/vkb*(logz - psi_m_unstable(zeta,zeta0))
            ustar1 = windspeed*vkb/(logz - psi_m_unstable(zeta,zeta0))
            lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * vkb)
            zeta = zb/lmo
            zeta0 = z0/lmo
            f1 = windspeed - ustar1/vkb*(logz - psi_m_unstable(zeta,zeta0))
            ustar = ustar1
            delta_ustar = ustar1 -ustar0
            while(np.abs(delta_ustar) > 1e-10):
                ustar_new = ustar1 - f1 * delta_ustar/(f1-f0)
                f0 = f1
                ustar0 = ustar1
                ustar1 = ustar_new
                lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * vkb)
                zeta = zb/lmo
                zeta0 = z0/lmo
                f1 = windspeed - ustar1/vkb*(logz - psi_m_unstable(zeta,zeta0))
                delta_ustar = ustar1 -ustar0
    else:
        ustar = ustar0

    return ustar



'''Wind Speed'''
cpdef void compute_windspeed(Grid Gr, double u, double v, double speed, double u0, double v0, double gustiness):
    cdef:
        # Py_ssize_t k
        # Py_ssize_t kmin = 1
        # Py_ssize_t kmax = Gr.nzg
        Py_ssize_t gw = Gr.gw

    u_ = u + u0
    v_ = v + v0
    speed = np.fmax(np.sqrt(u*u + v*v),gustiness)

    return
# cpdef void compute_windspeed(Grid Gr, double [:] u, double [:] v, double [:] speed, double u0, double v0, double gustiness):
#     cdef:
#         Py_ssize_t istride = 1#dims->nlg[1] * dims->nlg[2]
#         Py_ssize_t jstride = 1#dims->nlg[2]
#         Py_ssize_t istride_2d = 1#dims->nlg[1]
#
#         Py_ssize_t i, j, k
#         Py_ssize_t imin = 1
#         Py_ssize_t jmin = 1
#         Py_ssize_t kmin = 1
#         Py_ssize_t ishift, jshift
#         Py_ssize_t ij, ijk
#
#         Py_ssize_t imax = 1#dims->nlg[0]
#         Py_ssize_t jmax = 1#dims->nlg[1]
#         Py_ssize_t kmax = Gr.nzg
#         Py_ssize_t gw = Gr.gw
#
#     for i in xrange(imin, imax):
#         ishift = i*istride
#         for j in xrange(jmin, jmax):
#             jshift = j*jstride
#             ij = i * istride_2d + j
#             ijk = ishift + jshift + gw
#             u_interp = 0.5*(u[ijk-istride]+u[ijk]) + u0
#             v_interp = 0.5*(v[ijk-jstride]+v[ijk]) + v0
#             speed[ij] = np.fmax(np.sqrt(u_interp*u_interp + v_interp*v_interp),gustiness)
#
#     return


# cdef extern from "advection_interpolation.h":
#     double interp_2(double phi, double phip1) nogil
# cdef extern from "surface.h":
#     double compute_ustar(double windspeed, double buoyancy_flux, double z0, double z1) nogil
#     inline double entropyflux_from_thetaflux_qtflux(double thetaflux, double qtflux, double p0_b, double T_b, double qt_b, double qv_b) nogil
#     void compute_windspeed(Grid.DimStruct *dims, double* u, double*  v, double*  speed, double u0, double v0, double gustiness ) nogil
#     void exchange_coefficients_byun(double Ri, double zb, double z0, double* cm, double* ch, double* lmo) nogil
# cdef extern from "entropies.h":
#     inline double sd_c(double pd, double T) nogil
#     inline double sv_c(double pv, double T) nogil