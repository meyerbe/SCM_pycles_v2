import sys
from libc.math cimport fmin, fmax, sin
import cython
# import netCDF4 as nc
import numpy as np
cimport numpy as np

from Grid cimport Grid
from ReferenceState cimport ReferenceState
# cimport PrognosticVariables as PrognosticVariables
from PrognosticVariables cimport MeanVariables
from PrognosticVariables cimport SecondOrderMomenta
# cimport DiagnosticVariables

# from thermodynamic_functions cimport pd_c, pv_c
# from entropies cimport sv_c, sd_c

include 'parameters.pxi'

cdef class Damping:
    def __init__(self, namelist):
        if(namelist['damping']['scheme'] == 'None'):
            self.scheme = Dummy(namelist)
            print('No Damping!')
        elif(namelist['damping']['scheme'] == 'Rayleigh'):
            self.scheme = Rayleigh(namelist)
            print('Using Rayleigh Damping')
        return

    cpdef initialize(self, Grid Gr, ReferenceState RS):
        self.scheme.initialize(Gr, RS)
        return

    # cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV):
    cpdef update(self, Grid Gr, ReferenceState RS, MeanVariables M1, SecondOrderMomenta M2):
        # self.scheme.update(Gr, RS, PV, DV)
        self.scheme.update(Gr, RS, M1, M2)
        return


cdef class Dummy:
    def __init__(self, namelist):
        return

    cpdef initialize(self, Grid Gr, ReferenceState RS):
        return

    # cpdef update(self, Grid Gr, ReferenceState RS, PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV):
    cpdef update(self, Grid Gr, ReferenceState RS, MeanVariables M1, SecondOrderMomenta M2):
        return


cdef class Rayleigh:
    def __init__(self, namelist):
        try:
            self.z_d = namelist['damping']['Rayleigh']['z_d']
        except:
            print('Rayleigh damping z_d not given in namelist')
            print('Killing simulation now!')
            #Pa.kill()
            sys.exit()

        try:
            self.gamma_r = namelist['damping']['Rayleigh']['gamma_r']
        except:
            print('Rayleigh damping gamm_r not given in namelist')
            print('Killing simulation now!')
            #Pa.kill()
            sys.exit()

        return

    cpdef initialize(self, Grid Gr, ReferenceState RS):
#         cdef:
#             int k
#             double z_top
#
#         self.gamma_zhalf = np.zeros(
#             (Gr.dims.nlg[2]),
#             dtype=np.double,
#             order='c')
#         self.gamma_z = np.zeros((Gr.dims.nlg[2]), dtype=np.double, order='c')
#         z_top = Gr.dims.dx[2] * Gr.dims.n[2]
#         with nogil:
#             for k in range(Gr.dims.nlg[2]):
#                 if Gr.zl_half[k] >= z_top - self.z_d:
#                     self.gamma_zhalf[
#                         k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zl_half[k]) / self.z_d))**2.0
#                 if Gr.zl[k] >= z_top - self.z_d:
#                     self.gamma_z[
#                         k] = self.gamma_r * sin((pi / 2.0) * (1.0 - (z_top - Gr.zl[k]) / self.z_d))**2.0
        return

    # cpdef update(self, Grid Gr, ReferenceState.ReferenceState RS, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV):
    cpdef update(self, Grid Gr, ReferenceState RS, MeanVariables M1, SecondOrderMomenta M2):
#         cdef:
#             Py_ssize_t var_shift
#             Py_ssize_t imin = Gr.dims.gw
#             Py_ssize_t jmin = Gr.dims.gw
#             Py_ssize_t kmin = Gr.dims.gw
#             Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
#             Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
#             Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
#             Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
#             Py_ssize_t jstride = Gr.dims.nlg[2]
#             Py_ssize_t i, j, k, ishift, jshift, ijk
#             double[:] domain_mean
#
#         for var_name in PV.name_index:
#             var_shift = PV.get_varshift(Gr, var_name)
#             domain_mean = Pa.HorizontalMean(Gr, & PV.values[var_shift])
#             if var_name == 'w':
#                 with nogil:
#                     for i in xrange(imin, imax):
#                         ishift = i * istride
#                         for j in xrange(jmin, jmax):
#                             jshift = j * jstride
#                             for k in xrange(kmin, kmax):
#                                 ijk = ishift + jshift + k
#                                 PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_zhalf[k]
#             else:
#                 with nogil:
#                     for i in xrange(imin, imax):
#                         ishift = i * istride
#                         for j in xrange(jmin, jmax):
#                             jshift = j * jstride
#                             for k in xrange(kmin, kmax):
#                                 ijk = ishift + jshift + k
#                                 PV.tendencies[var_shift + ijk] -= (PV.values[var_shift + ijk] - domain_mean[k]) * self.gamma_z[k]
        return



# cdef class DampingToDomainMean:
#     def __init__(self,grid):
#         self.damping_depth = None
#         self.damping_timescale = None
#         self.do_damping = None
#         self.damping_zmin = None
#         self.damping_coefficient_center = None
#         self.damping_coefficient_interface = None
#
#     # cpdef initialize(self,case_dict,grid):
#     cpdef initialize(self, Grid Gr, ReferenceState RS):
#         try:
#            self.damping_depth = case_dict['damping']['depth']
#         except:
#             print('ERROR: Damping depth not specified in case_dict')
#             print('Killing Simulation')
#             sys.exit()
#
#         try:
#             self.damping_timescale = case_dict['damping']['timescale']
#         except:
#             print('ERROR: Damping timescale not specified in case_dict')
#             print('Killing Simulation Now')
#             sys.exit()
#
#
#         #Determine if it is necessary to do damping
#         self.do_damping = np.max(grid.zc) >= grid.zc_max - self.damping_depth
#
#         #Detarmine the lowest height for the damping
#         self.damping_zmin = grid.zc_max - self.damping_depth
#
#         #Now build damping factor
#         self.damping_coefficient_center =  np.zeros(grid.nzl,dtype=np.double,order='c')
#         self.damping_coefficient_interface   =  np.zeros(grid.nzl,dtype=np.double,order='c')
#
#
#         for k in xrange(grid.nzl):
#             if grid.zc[k] > self.damping_zmin:
#                 self.damping_coefficient_center[k] = np.maximum(((grid.zi_max - grid.zc[k])/(self.damping_depth)
#                                                              /self.damping_timescale),0.0)
#
#                 self.damping_coefficient_center[k] = np.maximum((1.0/self.damping_timescale - self.damping_coefficient_center[k]),
#                                                             0.0)
#             if grid.zi[k] > self.damping_zmin:
#                 self.damping_coefficient_interface[k] = np.maximum(((grid.zi_max - grid.zi[k])/(self.damping_depth)
#                                                             /self.damping_timescale),0.0)
#
#                 self.damping_coefficient_interface[k] = np.maximum(((1.0/self.damping_timescale - self.damping_coefficient_interface[k])),
#                                                            0.0)
#
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     @cython.cdivision(True)
#     # def update(self,grid,scalars,velocities):
#     # cpdef update(self, Grid Gr, ReferenceState RS, PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV):
#     cpdef update(self, Grid Gr, ReferenceState RS, MeanVariables M1, SecondOrderMomenta M2):
#         if not self.do_damping:
#             return
#
#         cdef int nxl = grid.nxl
#         cdef int nyl = grid.nyl
#         cdef int nzl = grid.nzl
#
#         #Get typed memory views on the velocities
#         cdef int number_momentum_dofs = velocities.ndof
#         cdef double [:,:,:,:] velocity_tendencies = velocities.tendencies[:,:,:,:]
#         cdef double [:,:,:,:] velocity_values = velocities.values[:,:,:,:]
#
#         #Get The Mean Profiles for Damping of Velocities
#         cdef double [:,:] velocity_means = velocities.mean_profiles
#
#         #Get the DOF for the w velocity component
#         cdef int wdof = velocities.get_dof('w')
#
#         #Get typed memory views on the scalars
#         cdef int number_scalar_dofs = scalars.ndof
#         cdef double [:,:,:,:] scalar_tendencies = scalars.tendencies[:,:,:,:]
#         cdef double [:,:,:,:] scalar_values = scalars.values[:,:,:,:]
#
#         #Get The Mean Profiles for Damping of Scalars
#         cdef double [:,:] scalar_means = scalars.mean_profiles
#
#         cdef double [:] damping_coefficient_center = self.damping_coefficient_center[:]
#         cdef double [:] damping_coefficient_interface = self.damping_coefficient_interface[:]
#
#         cdef int i,j,k,n
#         with nogil:
#             for i in prange(nxl):
#                 for j in xrange(nyl):
#                     for k in xrange(nzl):
#                         for n in xrange(number_momentum_dofs):
#                             if n != wdof:
#                                 velocity_tendencies[i,j,k,n] = velocity_tendencies[i,j,k,n]  - (
#                                     ((velocity_values[i,j,k,n] - velocity_means[k,n])
#                                      *damping_coefficient_center[k]))
#                             else:
#                                 velocity_tendencies[i,j,k,n] = velocity_tendencies[i,j,k,n] - (
#                                     ((velocity_values[i,j,k,n] - 0.0)
#                                      *damping_coefficient_interface[k]))
#
#                         for n in xrange(number_scalar_dofs):
#                             scalar_tendencies[i,j,k,n] = scalar_tendencies[i,j,k,n] - (
#                                 ((scalar_values[i,j,k,n] - scalar_means[k,n])*damping_coefficient_center[k]))
#
#
#         return