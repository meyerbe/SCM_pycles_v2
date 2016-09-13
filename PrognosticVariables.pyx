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
            Py_ssize_t k, var_shift
        for var in self.name_index.keys():
            var_shift = self.get_varshift(Gr, var)
            for k in xrange(0,kmax):
                self.values[var_shift+k] += self.tendencies[var_shift+k] * TS.dt

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
        self.velocity_directions = np.zeros((Gr.dims,),dtype=np.int,order='c')
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
        # print('u', self.name_index['u'])
        return


    cpdef initialize(self, Grid Gr, NetCDFIO_Stats NS):
        try:
            self.velocity_directions[0] = self.get_nv('u')
            self.velocity_directions[1] = self.get_nv('v')
            self.velocity_directions[2] = self.get_nv('w')
        except:
            print('problem setting velocity directions')
            print('Killing simulation now!')
            sys.exit()

        print('M1:', self.name_index)
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

        for var in xrange(self.nv):
            for k in xrange(0,kmax):
                self.values[var,k] += self.tendencies[var,k] * TS.dt
                self.tendencies[var,k] = 0.0

        # u_index = self.name_index['u']
        # print('M1: M1_tendencies[u,k=10]: ', self.tendencies[u_index+10], np.amax(self.tendencies))
        # print('M1: M1_tendencies[u,k=10]: ', self.tendencies[10], np.amax(self.tendencies))
        # th_varshift = self.get_varshift(Gr, 'th')
        # print('M1: M1_tendencies[phi=th,k=10]: ', self.tendencies[th_varshift+10], np.amax(self.tendencies))

        return


    cpdef plot(self, str message, Grid Gr, TimeStepping TS):
        print('!!! M1 plotting')
        cdef:
            double [:,:] values = self.values
            Py_ssize_t th_varshift = self.name_index['th']
            Py_ssize_t w_varshift = self.name_index['w']
            Py_ssize_t v_varshift = self.name_index['v']
            Py_ssize_t u_varshift = self.name_index['u']

        plt.figure(1,figsize=(15,7))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(values[th_varshift,:], Gr.z)
        plt.title('th')
        plt.subplot(1,4,2)
        plt.plot(values[w_varshift,:], Gr.z)
        plt.title('w')
        plt.subplot(1,4,3)
        plt.plot(values[v_varshift,:], Gr.z)
        plt.title('v')
        plt.subplot(1,4,4)
        plt.plot(values[u_varshift,:], Gr.z)
        plt.title('u')
        # plt.show()
        plt.savefig('./figs/M1_profiles_' + message + '_' + np.str(TS.t) + '.png')
        plt.close()



    cpdef plot_tendencies(self, str message, Grid Gr, TimeStepping TS):
        cdef:
            double [:,:] tendencies = self.tendencies
            Py_ssize_t th_varshift = self.name_index['th']
            Py_ssize_t w_varshift = self.name_index['w']
            Py_ssize_t v_varshift = self.name_index['v']
            Py_ssize_t u_varshift = self.name_index['u']
        plt.figure(2,figsize=(15,7))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(tendencies[th_varshift,:], Gr.z)
        plt.title('s tend')
        plt.subplot(1,4,2)
        plt.plot(tendencies[w_varshift,:], Gr.z)
        plt.title('w tend')
        plt.subplot(1,4,3)
        plt.plot(tendencies[v_varshift,:], Gr.z)
        plt.title('v tend')
        plt.subplot(1,4,4)
        plt.plot(tendencies[u_varshift,:], Gr.z)
        plt.title('u tend')
        # plt.show()
        plt.savefig('./figs/M1_tendencies_' + message + '_' + np.str(TS.t) + '.png')
        plt.close()
        return


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
        self.name_index = {}        # key = name of correlation, e.g. 'wu'
        self.index_name = []        # list of correlation names
        self.var_index = {}         # key = name of variables (u,v,w,th,qt,p)
        self.units = {}
        self.nv = 0
        self.nv_scalars = 0
        self.nv_velocities = 0
        self.var_type = np.array([],dtype=np.int,order='c')
        return


    cpdef add_variable(self,name,units,var_type,n,m):
        # Store names and units
        self.name_index[name] = [n,m]
        self.index_name.append(name)
        self.units[name] = units

        #Set the type of the variable being added
        # 0=velocity*velocity; 1=scalars*scalar or velocity*scalar; 2=pressure correlation
        if var_type == "velocity":
            self.var_type = np.append(self.var_type,0)
            self.nv_velocities += 1
        elif var_type == "scalar":
            self.var_type = np.append(self.var_type,1)
            self.nv_scalars += 1
        elif var_type == "pressure":
            self.var_type = np.append(self.var_type,2)
        else:
            print("Not a valid var_type. Killing simulation now!")
            sys.exit()

        return

    cpdef initialize(self, Grid Gr, MeanVariables M1, NetCDFIO_Stats NS):
        print('2nd order moments: ')
        print(M1.name_index)
        # self.var_index = M1.name_index
        # self.nv = len(self.var_index.keys())

        '''Local Covariances'''

        '''Momentum (Co)Variances: uu, uv, uw, vv, vw, ww'''
        for m in xrange(M1.nv_velocities):
            var1 = M1.index_name[m]
            self.var_index[var1] = self.nv
            self.nv = len(self.var_index.keys())
            for n in xrange(m,M1.nv_velocities):
                var2 = M1.index_name[n]
                # print('!!!', var1,n,var2,m)
                self.add_variable(var1+var2,'(m/s)^2',"velocity",m,n)
            '''Scalar Fluxes: wth, wqt'''
            for m in xrange(M1.nv_scalars):
                var2 = M1.index_name[M1.nv_velocities + m]
                unit = '(m/1)'+ M1.units[var2]
                self.add_variable(var1+var2,unit,"scalar",n,m)
            '''Pressure Correlation'''
            m = M1.nv
            self.add_variable(var1+'p','(m/s)(N/m)',"pressure",n,m)

        '''Scalar Variances and Covariances: thth, thqt, qtqt'''
        for n in xrange(M1.nv_scalars):
            var1 = M1.index_name[M1.nv_velocities + n]
            self.var_index[var1] = self.nv
            self.nv = len(self.var_index.keys())
            for m in xrange(n,M1.nv_scalars):
                var2 = M1.index_name[M1.nv_velocities + m]
                unit = M1.units[var1] + M1.units[var2]
                self.add_variable(var1+var2,unit,"scalar",n,m)
            m = M1.nv
            self.add_variable(var1+'p','(m/s)(N/m)',"pressure",n,m)
        self.var_index['p'] = self.nv
        self.nv = len(self.var_index.keys())


        # print('!! nv !!', self.nv, M1.nv)
        # print(self.var_index)
        # print(M1.name_index)
        # print('name_index', self.name_index)
        # print('index_name', self.index_name)

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


    cpdef plot(self, str message, Grid Gr, TimeStepping TS):
        print('!!! M1 plotting')
        cdef:
            double [:,:,:] values = self.values
            Py_ssize_t th_varshift = self.var_index['th']
            Py_ssize_t w_varshift = self.var_index['w']
            Py_ssize_t v_varshift = self.var_index['v']
            Py_ssize_t u_varshift = self.var_index['u']

        plt.figure(1,figsize=(15,7))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(values[th_varshift,th_varshift,:], Gr.z)
        plt.title('thth')
        plt.subplot(1,4,2)
        plt.plot(values[w_varshift,w_varshift,:], Gr.z)
        plt.title('ww')
        plt.subplot(1,4,3)
        plt.plot(values[w_varshift,u_varshift,:], Gr.z)
        plt.title('wu')
        plt.subplot(1,4,4)
        plt.plot(values[w_varshift,th_varshift,:], Gr.z)
        plt.title('wth')
        # plt.show()
        plt.savefig('./figs/M2_profiles_' + message + '_' + np.str(TS.t) + '.png')
        plt.close()



    cpdef plot_tendencies(self, str message, Grid Gr, TimeStepping TS):
        cdef:
            double [:,:,:] tendencies = self.tendencies
            Py_ssize_t th_varshift = self.var_index['th']
            Py_ssize_t w_varshift = self.var_index['w']
            Py_ssize_t v_varshift = self.var_index['v']
            Py_ssize_t u_varshift = self.var_index['u']
        plt.figure(2,figsize=(15,7))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(tendencies[th_varshift,th_varshift,:], Gr.z)
        plt.title('thth tend')
        plt.subplot(1,4,2)
        plt.plot(tendencies[w_varshift,w_varshift,:], Gr.z)
        plt.title('ww tend')
        plt.subplot(1,4,3)
        plt.plot(tendencies[w_varshift,u_varshift,:], Gr.z)
        plt.title('wu tend')
        plt.subplot(1,4,4)
        plt.plot(tendencies[w_varshift,th_varshift,:], Gr.z)
        plt.title('wth tend')
        # plt.show()
        plt.savefig('./figs/M2_tendencies_' + message + '_' + np.str(TS.t) + '.png')
        plt.close()
        return