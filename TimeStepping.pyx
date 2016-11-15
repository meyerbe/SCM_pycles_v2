#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

# cimport PrognosticVariables as PrognosticVariables
# # cimport DiagnosticVariables as DiagnosticVariables
# cimport Grid as Grid
# # cimport Restart

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

        try:
            self.plot_freq = namelist['visualization']['frequency']
        except:
            self.plot_freq = 60.0
        print('plotting frequency is: ', self.plot_freq)

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
        try:
            self.t = namelist['time_stepping']['t']
        except:
            print('t (initial time) not given in namelist so taking default value t = 0')
            self.t = 0.0

        # # try:
        # #     self.cfl_limit = namelist['time_stepping']['cfl_limit']
        # # except:
        # #     Pa.root_print('cfl_limit (maximum permissible cfl number) not given in namelist so taking default value cfl_max=0.7')
        # #     self.cfl_limit = 0.7

        return


    cpdef update(self):
        self.t += self.dt
    #     self.nstep += 1
    #     print('time:', self.t)
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



