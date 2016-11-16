import numpy as np
cimport numpy as np

from Grid cimport Grid

from thermodynamic_functions cimport pd_c, pv_c, exner, cpm_c
from thermodynamic_functions cimport theta_rho

'''Stability Functions'''
cpdef inline double psi_m_unstable(double zeta, double zeta0)
cpdef inline double psi_h_unstable(double zeta, double zeta0)
cpdef inline double psi_m_stable(double zeta, double zeta0)
cpdef inline double psi_h_stable(double zeta, double zeta0)

'''Friction Velocity'''
cpdef double compute_ustar(double windspeed, double buoyancy_flux, double z0, double zb)

'''Wind Speed'''
cpdef void compute_windspeed(Grid Gr, double u, double v, double speed, double u0, double v0, double gustiness)
# cpdef void compute_windspeed(Grid Gr, double [:] u, double [:] v, double [:] speed, double u0, double v0, double gustiness)