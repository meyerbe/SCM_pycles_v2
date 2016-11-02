#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
import sys
import pylab as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'xx-small',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-small',
         'ytick.labelsize':'xx-small'}
plt.rcParams.update(params)

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
        self.bc_type = np.array([],dtype=np.double,order='c')
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
        self.bc_type = np.array([],dtype=np.double,order='c')
        self.velocity_directions = np.zeros((Gr.dims,),dtype=np.int,order='c')
        return


    cpdef add_variable(self,name,units,bc_type,var_type):
    # cpdef add_variable(self,name,units,var_type):
        # Store names and units
        self.name_index[name] = self.nv
        self.index_name.append(name)
        self.units[name] = units
        self.nv = len(self.name_index.keys())

        #Add bc type to array
        if bc_type == "sym":
            self.bc_type = np.append(self.bc_type,[1.0])
        elif bc_type =="asym":
            self.bc_type = np.append(self.bc_type,[-1.0])
        else:
            print("Not a valid bc_type.")

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
        print('adding M1 Variable ', name, self.nv)
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
            NS.add_profile(var_name, Gr)
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




    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cpdef update_boundary_conditions(self, Grid Gr):
        print('Updating M1 BCS')
        cdef:
            Py_ssize_t nv = self.nv
            Py_ssize_t gw = Gr.gw
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t k, kstart
    #     cdef int ndof = self.ndof
    #     cdef int n = 0
    #     cdef int  _coords = comm.cart_comm.Get_coords(_rank)[2]
    #     cdef int  _mpi_kdim = comm.nprocs[2]
            double [:,:] values = self.values
            double [:] bcfactor = self.bc_type
            double [:,:] temp = self.values #np.zeros(shape=values.shape)

    #     cdef int kstart
    #     cdef int nzl = grid.nzl

    # (1) set bottom boundary condition
        w_varshift = self.name_index['w']
        plt.figure(figsize=(6,5))
        plt.subplot(1,2,1)
        # plt.plot(values[2,:],Gr.z,'-x')
        plt.plot(values[w_varshift,:],Gr.z,'g-')
        plt.plot(values[w_varshift,nzg-gw:nzg],Gr.z[nzg-gw:nzg],'rx')
        plt.plot(values[w_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.title('w before BC changes')
        plt.subplot(1,2,2)
        plt.plot(values[0,:],Gr.z,'-x')
        plt.plot(values[0,nzg-gw:nzg],Gr.z[nzg-gw:nzg],'rx')
        plt.plot(values[0,0:gw],Gr.z[0:gw],'rx')
        plt.title('u before BC changes')
        plt.savefig('figs/M1_profiles_beforeBC.pdf')
        # plt.show()
        plt.close()

        #     with nogil:
        if 1 == 1:
            kstart = gw
            for k in xrange(gw):
                for n in xrange(nv):
                    if (bcfactor[n] == 1):
                        #print(n, 'bcfactor=1', gw, k, kstart-1-k, kstart+k, bcfactor[n], values[n,kstart-1-k], values[n,kstart+k]*bcfactor[n])
                        values[n,kstart-1-k] = values[n,kstart+k]*bcfactor[n]
                    else:
                        if k==0:
                            #print(n, 'bcfactor= -1, k=0', gw, k, kstart-1-k, kstart+k, bcfactor[n], 0.0)
                            values[n,kstart-1-k] = 0.0
                        else:
                            #print(n, 'bcfactor= -1', gw, k, kstart-1-k, kstart+k, bcfactor[n], values[n,kstart-1-k], values[n,kstart+k]*bcfactor[n])
                            values[n,kstart-1-k] = values[n,kstart+k]*bcfactor[n]

        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(Gr.z,values[2,:],'-x')
        plt.plot(Gr.z[nzg-gw:nzg],values[2,nzg-gw:nzg],'rx')
        plt.plot(Gr.z[0:gw],values[2,0:gw],'rx')
        plt.title('w after bottom BC changes')
        plt.subplot(1,2,2)
        plt.plot(Gr.z,values[0,:],'-x')
        plt.plot(Gr.z[nzg-gw:nzg],values[0,nzg-gw:nzg],'rx')
        plt.plot(Gr.z[0:gw],values[0,0:gw],'rx')
        plt.title('u after bottom BC changes')
        plt.savefig('figs/M1_profiles_afterbottomBC.pdf')
        # plt.show()
        plt.close()

    # (2) set top boundary condition
    #         with nogil:
        if 1 == 1:
            kstart = nzg - gw
            for k in xrange(gw):
                for n in xrange(nv):
                    if(bcfactor[n] == 1):
                        values[n,kstart+k] = values[n,kstart-k-1] * bcfactor[n]
                    else:
                        if(k == 0):
                            values[n,kstart+k] = 0.0
                        else:
                            values[n,kstart+k] = values[n,kstart-k] * bcfactor[n]


        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(Gr.z,values[2,:],'-x')
        plt.plot(Gr.z[nzg-gw:nzg],values[2,nzg-gw:nzg],'rx')
        plt.plot(Gr.z[0:gw],values[2,0:gw],'rx')
        plt.title('w after top BC changes')
        plt.subplot(1,2,2)
        plt.plot(Gr.z,values[0,:],'-x')
        plt.plot(Gr.z[nzg-gw:nzg],values[0,nzg-gw:nzg],'rx')
        plt.plot(Gr.z[0:gw],values[0,0:gw],'rx')
        plt.title('u after top BC changes')
        plt.savefig('figs/M1_profiles_aftertopBC.pdf')
        # plt.show()
        return




    cpdef plot(self, str message, Grid Gr, TimeStepping TS):
        cdef:
            double [:,:] values = self.values
            Py_ssize_t th_varshift = self.name_index['th']
            Py_ssize_t w_varshift = self.name_index['w']
            Py_ssize_t v_varshift = self.name_index['v']
            Py_ssize_t u_varshift = self.name_index['u']
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t nz = Gr.nz
            Py_ssize_t gw = Gr.gw

        if np.isnan(values).any():
            print('!!!!!', message, ' NAN in M1')

        plt.figure(1,figsize=(12,5))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(values[th_varshift,:], Gr.z, '-x')
        plt.plot(values[th_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(values[th_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        plt.title('th, max=' + np.str(np.round(np.amax(values[th_varshift,:]),2)), fontsize=10)
        # plt.xlim(292.999,293.001)
        plt.subplot(1,4,2)
        # plt.plot(values[w_varshift,0:nz+2*gw-1], Gr.z[0:nz+2*gw-1], '-x')
        plt.plot(values[w_varshift,0:nzg], Gr.z[0:nzg], '-x')
        plt.plot(values[w_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(values[w_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        # plt.xlim(-1e-4,1e-4)
        plt.title('w, max=' + np.str(np.round(np.amax(values[w_varshift,0:]),2)), fontsize=10)
        plt.subplot(1,4,3)
        plt.plot(values[v_varshift,:], Gr.z)
        plt.plot(values[v_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(values[v_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        # plt.xlim(-1e-5,1e-4)
        plt.title('v, max=' + np.str(np.round(np.amax(values[v_varshift,:]),2)), fontsize=10)
        plt.subplot(1,4,4)
        plt.plot(values[u_varshift,:], Gr.z, '-x')
        plt.plot(values[u_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(values[u_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        # plt.xlim(-1e-5,1e-4)
        plt.title('u, max=' + np.str(np.round(np.amax(values[u_varshift,:]),2))+ ', ' + message, fontsize=10)
        plt.savefig('./figs/M1/M1_profiles_' + message + '_' + np.str(np.int(TS.t)) + '.pdf')
        # plt.show()
        plt.close()



    cpdef plot_tendencies(self, str message, Grid Gr, TimeStepping TS):

        cdef:
            double [:,:] tendencies = self.tendencies
            Py_ssize_t th_varshift = self.name_index['th']
            Py_ssize_t w_varshift = self.name_index['w']
            Py_ssize_t v_varshift = self.name_index['v']
            Py_ssize_t u_varshift = self.name_index['u']
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t nz = Gr.nz
            Py_ssize_t gw = Gr.gw
        if np.isnan(tendencies).any():
            print('!!!!!', message, ' NAN in M1 tendencies')

        plt.figure(2,figsize=(12,5))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(tendencies[th_varshift,:], Gr.z)
        plt.plot(tendencies[th_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(tendencies[th_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        plt.title('th tend, max=' + np.str(np.round(np.amax(tendencies[th_varshift,:]),2)), fontsize=10)
        plt.subplot(1,4,2)
        plt.plot(tendencies[w_varshift,:], Gr.z, '-x')
        plt.plot(tendencies[w_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(tendencies[w_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        # plt.xlim(-1e-4,1e-4)
        plt.title('w tend, max=' + np.str(np.round(np.amax(tendencies[w_varshift,:]),2)), fontsize=10)
        plt.subplot(1,4,3)
        plt.plot(tendencies[v_varshift,:], Gr.z, '-x')
        plt.plot(tendencies[v_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(tendencies[v_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        plt.title('v tend, max=' + np.str(np.round(np.amax(tendencies[v_varshift,:]),2)), fontsize=10)
        plt.subplot(1,4,4)
        plt.plot(tendencies[u_varshift,:], Gr.z, '-x')
        plt.plot(tendencies[u_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(tendencies[u_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        # plt.title('u tend, '+message)
        plt.title('u tend, max=' + np.str(np.round(np.amax(tendencies[u_varshift,:]),2))+ ', ' + message, fontsize=10)
        plt.savefig('./figs/M1/M1_tendencies_' + message + '_' + np.str(np.int(TS.t)) + '.pdf')
        # plt.show()
        plt.close()
        return

    cpdef get_variable_array(self,name,Grid Gr):
        index = self.name_index[name]
        view = np.array(self.values).view()
        # view.shape = (self.nv,Gr.dims.nlg[0],Gr.dims.nlg[1],Gr.dims.nlg[2])
        view.shape = (self.nv,Gr.nzg)
        return view[index,:]

    cpdef get_tendency_array(self,name,Grid Gr):
        index = self.name_index[name]
        view = np.array(self.tendencies).view()
        # view.shape = (self.nv,Gr.dims.nlg[0],Gr.dims.nlg[1],Gr.dims.nlg[2])
        view.shape = (self.nv,Gr.nzg)
        return view[index,:]












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
        self.nv = 0                 # M2.nv = M1.nv+1 (additional variable: pressure); M2.nv=len(M2.var_index)
        self.nv_scalars = 0
        self.nv_velocities = 0
        self.var_type = np.array([],dtype=np.int,order='c')
        self.bc_type = np.expand_dims(np.array([0],dtype=np.double,order='c'),axis=1,)
        return


    cpdef add_variable(self,name,units,bc_type,var_type,m,n):
        print('adding M2 Variable ', name, bc_type, var_type, units, m,n)
        # Store names and units
        self.name_index[name] = [m,n]
        self.index_name.append(name)
        self.units[name] = units

        s = self.bc_type.shape[0]
        if np.maximum(n,m)>(s-1):
            self.bc_type = np.append(self.bc_type,np.zeros((s,1)),axis=1)
            self.bc_type = np.append(self.bc_type,np.zeros((1,s+1)),axis=0)

        if bc_type == "sym":
            self.bc_type[m,n] = 1.0
        elif bc_type =="asym":
            self.bc_type[m,n] = -1.0
        else:
            print("Not a valid bc_type.")

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
        print('Initialize 2nd order moments: ')
        # self.var_index = M1.name_index
        # self.nv = len(self.var_index.keys())

        '''Local Covariances'''
        for m in xrange(M1.nv_velocities):
            '''Momentum (Co)Variances: uu, uv, uw, vv, vw, ww'''
            var1 = M1.index_name[m]
            self.var_index[var1] = self.nv
            self.nv = len(self.var_index.keys())
            for n in xrange(m,M1.nv_velocities):
                var2 = M1.index_name[n]
                # print('!!!', var1,n,var2,m)
                self.add_variable(var1+var2,'(m/s)^2',"sym","velocity",m,n)
            '''Scalar Fluxes: wth, wqt'''
            for n in xrange(M1.nv_scalars):
                var2 = M1.index_name[M1.nv_velocities + n]
                unit = '(m/1)'+ M1.units[var2]
                print('adding scalar fluxes: ', var1, var2, m, n+M1.nv_velocities)
                self.add_variable(var1+var2,unit,"sym","scalar",m,n+M1.nv_velocities)
            '''Pressure Correlation'''
            n = M1.nv
            self.add_variable(var1+'p','(m/s)(N/m)',"sym","pressure",m,n)

        '''Scalar Variances and Covariances: thth, thqt, qtqt'''
        for m in xrange(M1.nv_scalars):
            var1 = M1.index_name[M1.nv_velocities + m]
            self.var_index[var1] = self.nv
            self.nv = len(self.var_index.keys())
            for n in xrange(m,M1.nv_scalars):
                var2 = M1.index_name[M1.nv_velocities + n]
                unit = M1.units[var1] + M1.units[var2]
                self.add_variable(var1+var2,unit,"sym","scalar",m+M1.nv_velocities,n+M1.nv_velocities)
            n = M1.nv
            self.add_variable(var1+'p','(m/s)(N/m)',"sym","pressure",m+M1.nv_velocities,n)
        self.var_index['p'] = self.nv
        self.nv = len(self.var_index.keys())

        print('M2: nv=', self.nv, 'M1: nv=', M1.nv)
        print(self.var_index)
        print(M1.name_index)
        print('name_index', self.name_index)
        print('index_name', self.index_name)

        self.values = np.zeros((self.nv,self.nv,Gr.nzg),dtype=np.double,order='c')
        self.tendencies = np.zeros((self.nv,self.nv,Gr.nzg),dtype=np.double,order='c')

        # print('values:', self.values.shape, self.tendencies.shape, Gr.nzg)
        if np.isnan(self.values).any():
            print('!!! init: NANs in M2 values')
        if np.isnan(self.tendencies).any():
            print('!!! init: NANs in M2 tend')


        # Add prognostic variables to Statistics IO
        print('Setting up statistical output files PV.M2')
        for var_name in self.name_index.keys():
            print('M2: adding stats profiles', var_name)
            # Add mean profile
            NS.add_profile(var_name, Gr)

        print('M2.bc_type')
        print(np.array(self.bc_type))
        return


    cpdef update(self, Grid Gr, TimeStepping TS):
        cdef:
            Py_ssize_t kmax = Gr.nzg
            Py_ssize_t var1, var2

        for var1 in xrange(self.nv):
            for var2 in xrange(self.nv):
                for k in xrange(0,kmax):
                    self.values[var1,var2,k] += self.tendencies[var1,var2,k] * TS.dt
                    self.tendencies[var1,var2,k] = 0.0
        return


    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cpdef update_boundary_conditions(self, Grid Gr):
        print('Updating M2 BCS')
        cdef:
            Py_ssize_t nv = self.nv
            Py_ssize_t gw = Gr.gw
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t k, kstart
            double [:,:,:] values = self.values
            double [:,:] bcfactor = self.bc_type
            # double [:,:] bcfactor = np.ones((self.nv,self.nv))

            # double [:,:,:] temp = self.values #np.zeros(shape=values.shape)

    # (1) set bottom boundary condition
    #     w_varshift = self.name_index['w']
    #     plt.figure(figsize=(6,5))
    #     plt.subplot(1,2,1)
    #     # plt.plot(values[2,:],Gr.z,'-x')
    #     plt.plot(values[w_varshift,:],Gr.z,'g-')
    #     plt.plot(values[w_varshift,nzg-gw:nzg],Gr.z[nzg-gw:nzg],'rx')
    #     plt.plot(values[w_varshift,0:gw],Gr.z[0:gw],'rx')
    #     plt.title('w before BC changes')
    #     plt.subplot(1,2,2)
    #     plt.plot(values[0,:],Gr.z,'-x')
    #     plt.plot(values[0,nzg-gw:nzg],Gr.z[nzg-gw:nzg],'rx')
    #     plt.plot(values[0,0:gw],Gr.z[0:gw],'rx')
    #     plt.title('u before BC changes')
    #     plt.savefig('figs/M1_profiles_beforeBC.pdf')
    #     # plt.show()
    #     plt.close()

        #     with nogil:
        if 1 == 1:
            kstart = gw
            for k in xrange(gw):
                for m in xrange(nv):
                    for n in xrange(m,nv):
                        if (bcfactor[m,n] == 1):
                            print('m,n:',m,n, 'bcfactor=1', gw, k, bcfactor[m,n], values[m,n,kstart-1-k], values[m,n,kstart+k]*bcfactor[m,n])
                            values[m,n,kstart-1-k] = values[m,n,kstart+k]*bcfactor[m,n]
                        else:
                            if k==0:
                                print('m,n:',m,n, 'bcfactor= -1, k=0', gw, k, bcfactor[m,n], 0.0)
                                values[m,n,kstart-1-k] = 0.0
                            else:
                                print('m,n:',m,n, 'bcfactor= -1', gw, k, bcfactor[m,n], values[m,n,kstart-1-k], values[m,n,kstart+k]*bcfactor[m,n])
                                values[m,n,kstart-1-k] = values[m,n,kstart+k]*bcfactor[m,n]

        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.plot(Gr.z,values[2,:],'-x')
        # plt.plot(Gr.z[nzg-gw:nzg],values[2,nzg-gw:nzg],'rx')
        # plt.plot(Gr.z[0:gw],values[2,0:gw],'rx')
        # plt.title('w after bottom BC changes')
        # plt.subplot(1,2,2)
        # plt.plot(Gr.z,values[0,:],'-x')
        # plt.plot(Gr.z[nzg-gw:nzg],values[0,nzg-gw:nzg],'rx')
        # plt.plot(Gr.z[0:gw],values[0,0:gw],'rx')
        # plt.title('u after bottom BC changes')
        # plt.savefig('figs/M1_profiles_afterbottomBC.pdf')
        # # plt.show()
        # plt.close()

    # (2) set top boundary condition
    #         with nogil:
        if 1 == 1:
            kstart = nzg - gw
            for k in xrange(gw):
                for m in xrange(nv):
                    for n in xrange(nv):
                        if(bcfactor[m,n] == 1):
                            values[m,n,kstart+k] = values[m,n,kstart-k-1] * bcfactor[m,n]
                        else:
                            if(k == 0):
                                values[n,kstart+k] = 0.0
                            else:
                                values[m,n,kstart+k] = values[m,n,kstart-k] * bcfactor[m,n]


        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.plot(Gr.z,values[2,:],'-x')
        # plt.plot(Gr.z[nzg-gw:nzg],values[2,nzg-gw:nzg],'rx')
        # plt.plot(Gr.z[0:gw],values[2,0:gw],'rx')
        # plt.title('w after top BC changes')
        # plt.subplot(1,2,2)
        # plt.plot(Gr.z,values[0,:],'-x')
        # plt.plot(Gr.z[nzg-gw:nzg],values[0,nzg-gw:nzg],'rx')
        # plt.plot(Gr.z[0:gw],values[0,0:gw],'rx')
        # plt.title('u after top BC changes')
        # plt.savefig('figs/M1_profiles_aftertopBC.pdf')
        # # plt.show()
        return



    cpdef plot(self, str message, Grid Gr, TimeStepping TS):
        cdef:
            double [:,:,:] values = self.values
            Py_ssize_t th_varshift = self.var_index['th']
            Py_ssize_t w_varshift = self.var_index['w']
            Py_ssize_t v_varshift = self.var_index['v']
            Py_ssize_t u_varshift = self.var_index['u']
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t nz = Gr.nz
            Py_ssize_t gw = Gr.gw
        if np.isnan(values).any():
            print('!!!!! NAN in M2')

        plt.figure(1,figsize=(12,5))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(values[th_varshift,th_varshift,:], Gr.z)
        plt.plot(values[th_varshift,th_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(values[th_varshift,th_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        plt.title('thth, max:' + np.str(np.round(np.amax(values[th_varshift,th_varshift:]),2)), fontsize=10)
        plt.subplot(1,4,2)
        plt.plot(values[w_varshift,w_varshift,:], Gr.z)
        plt.plot(values[w_varshift,w_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(values[w_varshift,w_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        plt.title('ww, max:' + np.str(np.round(np.amax(values[w_varshift,w_varshift:]),2)), fontsize=10)
        plt.subplot(1,4,3)
        plt.plot(values[u_varshift,w_varshift,:], Gr.z)
        plt.plot(values[u_varshift,w_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(values[u_varshift,w_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        plt.title('uw, max:' + np.str(np.round(np.amax(values[u_varshift,w_varshift:]),2)), fontsize=10)
        plt.subplot(1,4,4)
        plt.plot(values[w_varshift,th_varshift,:], Gr.z)
        plt.plot(values[w_varshift,th_varshift,0:gw],Gr.z[0:gw],'rx')
        plt.plot(values[w_varshift,th_varshift,gw+nz:nzg],Gr.z[gw+nz:nzg],'rx')
        plt.title('wth, max:' + np.str(np.round(np.amax(values[w_varshift,th_varshift:]),2))+ ', ' + message, fontsize=10 )
        plt.savefig('./figs/M2_profiles_' + message + '_' + np.str(np.int(TS.t)) + '.pdf')
        # plt.show()
        plt.close()

        return



    cpdef plot_tendencies(self, str message, Grid Gr, TimeStepping TS):
        cdef:
            double [:,:,:] tendencies = self.tendencies
            Py_ssize_t th_varshift = self.var_index['th']
            Py_ssize_t w_varshift = self.var_index['w']
            Py_ssize_t v_varshift = self.var_index['v']
            Py_ssize_t u_varshift = self.var_index['u']

        if np.isnan(tendencies).any():
            print('!!!!! NAN in M2 tendencies')
        plt.figure(2,figsize=(12,5))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,4,1)
        plt.plot(tendencies[th_varshift,th_varshift,:], Gr.z, '-x')
        plt.plot(tendencies[th_varshift,th_varshift,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(tendencies[th_varshift,th_varshift,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        plt.title('thth tend')
        plt.subplot(1,4,2)
        plt.plot(tendencies[w_varshift,w_varshift,:], Gr.z, '-x')
        plt.plot(tendencies[w_varshift,w_varshift,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(tendencies[w_varshift,w_varshift,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        plt.title('ww tend')
        plt.subplot(1,4,3)
        plt.plot(tendencies[u_varshift,w_varshift,:], Gr.z, '-x')
        plt.plot(tendencies[u_varshift,w_varshift,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(tendencies[u_varshift,w_varshift,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        plt.title('wu tend')
        plt.subplot(1,4,4)
        plt.plot(tendencies[w_varshift,th_varshift,:], Gr.z, '-x')
        plt.plot(tendencies[w_varshift,th_varshift,0:Gr.gw], Gr.z[0:Gr.gw], 'rx')
        plt.plot(tendencies[w_varshift,th_varshift,Gr.gw+Gr.nz:Gr.nzg], Gr.z[Gr.gw+Gr.nz:Gr.nzg], 'rx')
        plt.title('wth tend, '+message)
        plt.savefig('./figs/M2_tendencies_' + message + '_' + np.str(np.int(TS.t)) + '.pdf')
        # plt.show()
        plt.close()
        return


    cpdef get_variable_array(self,name,Grid Gr):
        index = self.name_index[name]
        m = index[0]
        n = index[1]
        view = np.array(self.values).view()
        # view.shape = (self.nv,Gr.dims.nlg[0],Gr.dims.nlg[1],Gr.dims.nlg[2])
        view.shape = (self.nv,self.nv,Gr.nzg)
        return view[m,n,:]

    cpdef get_tendency_array(self,name,Grid Gr):
        # index = self.name_index[name]
        index = self.name_index[name]
        m = index[0]
        n = index[1]
        view = np.array(self.tendencies).view()
        # view.shape = (self.nv,Gr.dims.nlg[0],Gr.dims.nlg[1],Gr.dims.nlg[2])
        view.shape = (self.nv,self.nv,Gr.nzg)
        return view[m,n,:]