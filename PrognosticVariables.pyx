#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
import sys
import pylab as plt

from NetCDFIO cimport NetCDFIO_Stats
from Grid cimport Grid
from TimeStepping cimport TimeStepping

'''
self.name_index[str name]:      returns index of variable of given name
self.index_name[int i]:         returns name of given index
self.units[str name]:           returns unit of variable of given name
self.nv:                        number of variables
self.nv_scalars:                number of scalars
self.nv_velocities:             number of velocities
self.var_type[int i]:           type of variable (velocity==0, scalar==1)
self.velocity_directions[int dir]:   returns index of velocity of given direction dir (important to change from 3d to 2d or 1d dynamics
'''


cdef class PrognosticVariables:
    def __init__(self, Grid Gr):
        self.name_index = {}
        self.index_name = []
        self.units = {}
        self.nv = 0
        self.nv_scalars = 0
        self.nv_velocities = 0
        self.var_type = np.array([],dtype=np.int,order='c')
        return

    cpdef add_variable(self,name,units,var_type):
        # Store names and units
        self.name_index[name] = self.nv
        self.index_name.append(name)
        self.units[name] = units
        self.nv = len(self.name_index.keys())
        #Set the type of the variable being added 0=velocity; 1=scalars
        if var_type == "velocity":
            self.var_type = np.append(self.var_type,0)
            self.nv_velocities += 1
        elif var_type == "scalar":
            self.var_type = np.append(self.var_type,1)
            self.nv_scalars += 1
        else:
            print("Not a valid var_type. Killing simulation now!")
            sys.exit()
        print('adding Variable ', name, self.nv)
        # try:
        #     print(self.get_nv('u'))
        #     # self.velocity_directions[0] = self.get_nv('u')
        #     # self.velocity_directions[1] = self.get_nv('v')
        #     # self.velocity_directions[2] = self.get_nv('w')
        # except:
        #     print('problem setting velocity')
        #     print('Killing simulation now!')
        #     sys.exit()
        return

    cpdef initialize(self, Grid Gr, NetCDFIO_Stats NS):
    # cpdef initialize(self, Grid Gr):
        self.values = np.zeros((self.nv*Gr.nzg),dtype=np.double,order='c')
        self.tendencies = np.zeros((self.nv*Gr.nzg),dtype=np.double,order='c')
        # Add prognostic variables to Statistics IO
        print('Setting up statistical output files for Prognostic Variables')
        for var_name in self.name_index.keys():
            # Add mean profile
            NS.add_profile(var_name+'_mean', Gr)

            if var_name == 'u' or var_name == 'v':
                NS.add_profile(var_name+'_translational_mean', Gr)

            #Add max ts
            NS.add_ts(var_name+'_max',Gr)
            #Add min ts
            NS.add_ts(var_name+'_min',Gr)

        # if 'qt' in self.name_index.keys() and 's' in self.name_index.keys():
        #     NS.add_profile('qt_s_product_mean', Gr)
        return

    cpdef update(self, Grid Gr, TimeStepping TS):
        cdef:
            Py_ssize_t kmax = Gr.nzg
            Py_ssize_t k
        for var in self.name_index.keys():
            var_shift = self.get_varshift(Gr, var)
            for k in xrange(0,kmax):
                self.values[var_shift + k] += self.tendencies[var_shift + k] * TS.dt

        return




cdef class MeanVariables:
    def __init__(self, Grid Gr):
        self.name_index = {}
        self.index_name = []
        self.units = {}
        self.nv = 0
        self.nv_scalars = 0
        self.nv_velocities = 0
        self.var_type = np.array([],dtype=np.int,order='c')
        return

    cpdef add_variable(self,name,units,var_type):
        # Store names and units
        self.name_index[name] = self.nv
        self.index_name.append(name)
        self.units[name] = units
        self.nv = len(self.name_index.keys())
        #Set the type of the variable being added 0=velocity; 1=scalars
        if var_type == "velocity":
            self.var_type = np.append(self.var_type,0)
            self.nv_velocities += 1
        elif var_type == "scalar":
            self.var_type = np.append(self.var_type,1)
            self.nv_scalars += 1
        else:
            print("Not a valid var_type. Killing simulation now!")
            sys.exit()
        print('adding Variable ', name, self.nv)
        print('u', self.name_index['u'])
        return

    cpdef initialize(self, Grid Gr, NetCDFIO_Stats NS):
    # cpdef initialize(self, Grid Gr):
        # try:
        #     self.velocity_directions[0] = self.get_nv('u')      # Causes Problems!!!
        #     self.velocity_directions[1] = self.get_nv('v')
        #     self.velocity_directions[2] = self.get_nv('w')
        # except:
        #     print('problem setting velocity directions')
        #     print('Killing simulation now!')
        #     sys.exit()
        self.values = np.zeros(shape=(self.nv,Gr.nzg),dtype=np.double,order='c')
        self.tendencies = np.zeros(shape=(self.nv,Gr.nzg),dtype=np.double,order='c')
        # Add prognostic variables to Statistics IO
        print('Setting up statistical output files for PV.M1')
        for var_name in self.name_index.keys():
            #Add mean profile
            NS.add_profile(var_name+'_mean', Gr)
        return

    cpdef update(self, Grid Gr, TimeStepping TS):
        cdef:
            kmax = Gr.nzg
            Py_ssize_t var

        print('M1 shape', self.values.shape)
        for var in xrange(self.nv):
            print('var', var, self.nv)
            for k in xrange(0,kmax):
                self.values[var,k] += self.tendencies[var,k] * TS.dt
                self.tendencies[var,k] = 0.0


        u_index = self.name_index['u']
        print('M1: M1_tendencies[u,k=10]: ', self.tendencies[u_index+10], np.amax(self.tendencies))
        # print('M1: M1_tendencies[u,k=10]: ', self.tendencies[10], np.amax(self.tendencies))
        # th_varshift = self.get_varshift(Gr, 'th')
        # print('M1: M1_tendencies[phi=th,k=10]: ', self.tendencies[th_varshift+10], np.amax(self.tendencies))

        return


    # cpdef plot(self, str message, Grid Gr, TimeStepping TS):
    #     cdef:
    #         double [:,:] values = self.values
    #         double [:,:] tendencies = self.tendencies
    #         Py_ssize_t th_varshift = self.get_varshift(Gr,'th')
    #         Py_ssize_t w_varshift = self.get_varshift(Gr,'w')
    #         Py_ssize_t v_varshift = self.get_varshift(Gr,'v')
    #         Py_ssize_t u_varshift = self.get_varshift(Gr,'u')
    #
    #     plt.figure(1,figsize=(15,7))
    #     # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
    #     plt.subplot(1,4,1)
    #     plt.plot(values[th_varshift:th_varshift+Gr.nzg], Gr.z)
    #     plt.title('th')
    #     plt.subplot(1,4,2)
    #     plt.plot(values[w_varshift:w_varshift+Gr.nzg], Gr.z)
    #     plt.title('w')
    #     plt.subplot(1,4,3)
    #     plt.plot(values[v_varshift:v_varshift+Gr.nzg], Gr.z)
    #     plt.title('v')
    #     plt.subplot(1,4,4)
    #     plt.plot(values[u_varshift:u_varshift+Gr.nzg], Gr.z)
    #     plt.title('u')
    #     # plt.show()
    #     plt.savefig('./figs/profiles_' + message + '_' + np.str(TS.t) + '.png')
    #     plt.close()
    #
    #     plt.figure(2,figsize=(15,7))
    #     # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
    #     plt.subplot(1,4,1)
    #     plt.plot(tendencies[th_varshift:th_varshift+Gr.nzg], Gr.z)
    #     plt.title('s tend')
    #     plt.subplot(1,4,2)
    #     plt.plot(tendencies[w_varshift:w_varshift+Gr.nzg], Gr.z)
    #     plt.title('w tend')
    #     plt.subplot(1,4,3)
    #     plt.plot(tendencies[v_varshift:v_varshift+Gr.nzg], Gr.z)
    #     plt.title('v tend')
    #     plt.subplot(1,4,4)
    #     plt.plot(tendencies[u_varshift:u_varshift+Gr.nzg], Gr.z)
    #     plt.title('u tend')
    #     # plt.show()
    #     plt.savefig('./figs/tendencies_' + message + '_' + np.str(TS.t) + '.png')
    #     plt.close()
    #     return
    #
    # # # cpdef plot_tendencies(self, Grid Gr, TimeStepping TS):
    # # #     cdef:
    # # #         double [:] values = self.values
    # # #         double [:] tendencies = self.tendencies
    # # #         Py_ssize_t s_varshift = self.get_varshift(Gr,'s')
    # # #         Py_ssize_t w_varshift = self.get_varshift(Gr,'w')
    # # #         Py_ssize_t v_varshift = self.get_varshift(Gr,'v')
    # # #         Py_ssize_t u_varshift = self.get_varshift(Gr,'u')
    # # #     plt.figure(1,figsize=(15,7))
    # # #     # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
    # # #     plt.subplot(1,4,1)
    # # #     plt.plot(values[s_varshift:s_varshift+Gr.nzg], Gr.z)
    # # #     plt.title('s')
    # # #     plt.subplot(1,4,2)
    # # #     plt.plot(values[w_varshift:w_varshift+Gr.nzg], Gr.z)
    # # #     plt.title('w')
    # # #     plt.subplot(1,4,3)
    # # #     plt.plot(values[v_varshift:v_varshift+Gr.nzg], Gr.z)
    # # #     plt.title('v')
    # # #     plt.subplot(1,4,4)
    # # #     plt.plot(values[u_varshift:u_varshift+Gr.nzg], Gr.z)
    # # #     plt.title('u')
    # # #     plt.show()
    # # #     plt.savefig('./figs/profiles_' + np.str(TS.t) + '.png')
    # # #     plt.close()
    # # #     return








cdef class SecondOrderMomenta:
    # implementation for staggered grid
        # w: on w-grid
        # u,v,{s,qt}: on phi-grid
        # —> dz ws, dz wqt on phi-grid      —> ws, wqt on w-grid   -> compare to scalar advection for gradients
        # —> dz wu, dz wv on phi-grid       —> wu, wv on w-grid    -> compare to scalar advection for gradients
        # —> dz ww on w-grid                —> ww on phi-grid      -> compare to momentum advection for gradients

    def __init__(self, Grid Gr):
        self.name_index = {}
        self.index_name = []
        self.units = {}
        self.nv = 0
        self.nv_scalars = 0
        self.nv_velocities = 0
        self.var_type = np.array([],dtype=np.int,order='c')
        return

    cpdef add_variable(self,name,units,var_type):
        # Store names and units
        self.name_index[name] = self.nv
        self.index_name.append(name)
        self.units[name] = units
        self.nv = len(self.name_index.keys())
        #Set the type of the variable being added 0=velocity; 1=scalars
        if var_type == "velocity":
            self.var_type = np.append(self.var_type,0)
            self.nv_velocities += 1
        elif var_type == "scalar":
            self.var_type = np.append(self.var_type,1)
            self.nv_scalars += 1
        else:
            print("Not a valid var_type. Killing simulation now!")
            sys.exit()
        print('adding Variable ', name, self.nv)

        return

    cpdef initialize(self, Grid Gr, NetCDFIO_Stats NS):
        print('2nd order moments: ', self.nv)
        self.values = np.zeros((self.nv,self.nv,Gr.nzg),dtype=np.double,order='c')
        self.tendencies = np.zeros((self.nv,self.nv,Gr.nzg),dtype=np.double,order='c')

        # Add prognostic variables to Statistics IO
        print('Setting up statistical output files PV.M2')
        for var_name in self.name_index.keys():
            # Add mean profile
            NS.add_profile(var_name, Gr)
        return

    cpdef update(self, Grid Gr, TimeStepping TS):
        cdef:
            Py_ssize_t kmax = Gr.nzg
            Py_ssize_t var1, var2

        for var1 in xrange(self.nv):
            for var2 in xrange(self.nv):
                for k in xrange(0,kmax):
                    self.values[var1,var2,k] += self.tendencies[var1,var2,k] * TS.dt
        return