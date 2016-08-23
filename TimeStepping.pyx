#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport PrognosticVariables as PrognosticVariables
# cimport DiagnosticVariables as DiagnosticVariables
cimport Grid as Grid
# cimport Restart

import numpy as np
cimport numpy as np
import sys

from libc.math cimport fmin, fmax, fabs

cdef class TimeStepping:
    def __init__(self):

        return

    # cpdef initialize(self,namelist,PrognosticVariables.PrognosticVariables PV):
    cpdef initialize(self,namelist):
        print('Timestepping initialize')
        try:
            self.dt = namelist['time_stepping']['dt_initial']
        except:
            print('dt_initial (initial time step) not given in namelist so taking defualt value dt_initial = 100.0s')
            self.dt = 100.0

        try:
            self.t_max = namelist['time_stepping']['t_max']
        except:
            print('t_max (time at end of simulation) not given in name list! Killing Simulation Now')
            sys.exit()

        # set time
        self.t = 0.0
        self.nstep = 0

        #     #Initialize storage
        # self.value_copies = np.zeros((1,PV.values.shape[0]),dtype=np.double,order='c')
        # self.tendency_copies = None

        # try:
        #     self.dt_max = namelist['time_stepping']['dt_max']
        # except:
        #     Pa.root_print('dt_max (maximum permissible time step) not given in namelist so taking default value dt_max =10.0')
        #     self.dt_max = 10.0
        #
        # try:
        #     self.t = namelist['time_stepping']['t']
        # except:
        #     Pa.root_print('t (initial time) not given in namelist so taking default value t = 0')
        #     self.t = 0.0
        #
        # try:
        #     self.cfl_limit = namelist['time_stepping']['cfl_limit']
        # except:
        #     Pa.root_print('cfl_limit (maximum permissible cfl number) not given in namelist so taking default value cfl_max=0.7')
        #     self.cfl_limit = 0.7

        return


    # cpdef update(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV):
    cpdef update(self):
        self.t += self.dt
        self.nstep += 1
        print('time:', self.t)
    #     cdef:
    #         Py_ssize_t i
    #
    #     with nogil:
    #         if self.rk_step == 0:
    #             for i in xrange(Gr.dims.npg*PV.nv):
    #                 self.value_copies[0,i] = PV.values[i]
    #                 PV.values[i] += PV.tendencies[i]*self.dt
    #                 PV.tendencies[i] = 0.0
    #         else:
    #             for i in xrange(Gr.dims.npg*PV.nv):
    #                 PV.values[i] = 0.5 * (self.value_copies[0,i] + PV.values[i] + PV.tendencies[i] * self.dt)
    #                 PV.tendencies[i] = 0.0

        return





    # cpdef adjust_timestep(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, ParallelMPI.ParallelMPI Pa):
    #     #Compute the CFL number and diffusive stability criterion
    #     if self.rk_step == self.n_rk_steps - 1:
    #         self.compute_cfl_max(Gr, PV,DV, Pa)
    #         self.dt = self.cfl_time_step()
    #
    #         #Diffusive limiting not yet implemented
    #         if self.t + self.dt > self.t_max:
    #             self.dt = self.t_max - self.t
    #
    #         if self.dt < 0.0:
    #             Pa.root_print('dt = '+ str(self.dt)+ " killing simulation!")
    #             Pa.kill()
    #
    #     return


    #
    # cdef void compute_cfl_max(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV,DiagnosticVariables.DiagnosticVariables DV):
    #     Pa.root_print('Computing CFL Max')
    #     cdef:
    #         double cfl_max_local = -9999.0
    #         double [3] dxi = Gr.dims.dxi
    #         Py_ssize_t u_shift = PV.get_varshift(Gr,'u')
    #         Py_ssize_t v_shift = PV.get_varshift(Gr,'v')
    #         Py_ssize_t w_shift = PV.get_varshift(Gr,'w')
    #         Py_ssize_t imin = Gr.dims.gw
    #         Py_ssize_t jmin = Gr.dims.gw
    #         Py_ssize_t kmin = Gr.dims.gw
    #         Py_ssize_t imax = Gr.dims.nlg[0] - Gr.dims.gw
    #         Py_ssize_t jmax = Gr.dims.nlg[1] - Gr.dims.gw
    #         Py_ssize_t kmax = Gr.dims.nlg[2] - Gr.dims.gw
    #         Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
    #         Py_ssize_t jstride = Gr.dims.nlg[2]
    #         Py_ssize_t i,j,k, ijk, ishift, jshift
    #         double w
    #         Py_ssize_t isedv
    #
    #     with nogil:
    #         for i in xrange(imin,imax):
    #             ishift = i * istride
    #             for j in xrange(jmin,jmax):
    #                 jshift = j * jstride
    #                 for k in xrange(kmin,kmax):
    #                     ijk = ishift + jshift + k
    #                     w = fabs(PV.values[w_shift+ijk])
    #                     for isedv in xrange(DV.nsedv):
    #                         w = fmax(fabs( DV.values[DV.sedv_index[isedv]*Gr.dims.npg + ijk ] + PV.values[w_shift+ijk]), w)
    #
    #                     cfl_max_local = fmax(cfl_max_local, self.dt * (fabs(PV.values[u_shift + ijk])*dxi[0] + fabs(PV.values[v_shift+ijk])*dxi[1] + w*dxi[2]))
    #                     # problem: second term is nan
    #     Pa.root_print('cfl_max_local: '+ str(cfl_max_local))
    #     Pa.root_print(str(self.dt * (fabs(PV.values[u_shift + ijk])*dxi[0] + fabs(PV.values[v_shift+ijk])*dxi[1] + w*dxi[2])))  # is a nan
    #     # Pa.root_print('u: '+str(np.amax(PV.values[u_shift:v_shift])) + ', '+ str(np.amin(PV.values[u_shift:v_shift])))
    #     # Pa.root_print('v: '+str(np.amax(PV.values[v_shift:w_shift])) + ', '+ str(np.amin(PV.values[v_shift:w_shift])))
    #     # Pa.root_print('w: '+str(w))
    #
    #     mpi.MPI_Allreduce(&cfl_max_local,&self.cfl_max,1,
    #                       mpi.MPI_DOUBLE,mpi.MPI_MAX,Pa.comm_world)
    #
    #     self.cfl_max += 1e-11
    #
    #     if self.cfl_max < 0.0:
    #         Pa.root_print('CFL_MAX = '+ str(self.cfl_max)+ " killing simulation!")
    #         Pa.kill()
    #     return
    #
    # cdef inline double cfl_time_step(self):
    #     return fmin(self.dt_max,self.cfl_limit/(self.cfl_max/self.dt))
    #
    # cpdef restart(self, Restart.Restart Re):
    #     Re.restart_data['TS'] = {}
    #     Re.restart_data['TS']['t'] = self.t
    #     Re.restart_data['TS']['dt'] = self.dt
    #
    #     return
