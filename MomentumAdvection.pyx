#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from Grid cimport Grid
from TimeStepping cimport TimeStepping
from PrognosticVariables cimport MeanVariables
from PrognosticVariables cimport SecondOrderMomenta
from ReferenceState cimport ReferenceState
# from NetCDFIO cimport NetCDFIO_Stats

import numpy as np
cimport numpy as np
import sys
import pylab as plt


cdef class MomentumAdvection:
    def __init__(self, namelist):
        try:
            self.order = namelist['momentum_transport']['order']
            print('momentum transport order: ' + np.str(self.order))
        except:
            print('momentum_transport order not given in namelist')
            print('Killing simulation now!')
            sys.exit()

        return

    cpdef initialize(self, Grid Gr, MeanVariables M1):
        # initialize scheme for Mean Variables
        self.flux = np.zeros((M1.nv_velocities,Gr.nzg),dtype=np.double,order='c')
        self.tendencies = np.zeros((M1.nv_velocities,Gr.nzg),dtype=np.double,order='c')

        return


    cpdef update(self, Grid Gr, ReferenceState Ref, MeanVariables M1):
        # (1) update tendencies for Mean Variables
        #       - only vertical advection
        # dt U = ... - alpha0 dz (rho0 <w'u'>)
        # dt V = ... - alpha0 dz (rho0 <w'v'>)
        # dt W = ... - alpha0 dz (rho0 <w'w'>)
        # M1.tendencies += M2.values

        if self.order == 2:
            self.update_M1_2nd(Gr, Ref, M1)
        else:
            print('momentum advection scheme not implemented')
            sys.exit()

        # (2) update tendencies for 2nd Order Momenta
        # if self.order == 2:
        #     self.update_M2_2nd(Gr, Ref, M2)
        # else:
        #     print('momentum advection scheme not implemented')
        #     sys.exit()
        return




    cpdef update_M1_2nd(self, Grid Gr, ReferenceState Ref, MeanVariables M1):
        print('Momentum Advection M1: update 2nd')
        # (1) update tendencies for Mean Variables
        #       - only vertical advection
        # (1a) advection by mean velocity: 1/rho0*\partialz(rho0 <w><u_i>)
        # (1b) turbulent advection: 1/rho0*\partialz(<rho0 w'u'_i>)
        cdef:
            Py_ssize_t d_advected       # Direction of advected momentum component
            Py_ssize_t w_index = M1.name_index['w']

            Py_ssize_t k
            Py_ssize_t gw = Gr.gw
            double dzi = Gr.dzi
            double dzi2 = 0.5*Gr.dzi

            # Py_ssize_t stencil[3] = {istride,jstride,1}
            Py_ssize_t sp1_ed = 1
            Py_ssize_t sm1_ed = -sp1_ed

            # double [:] vel_advecting = M1.values[w_index,:]
            # double [:] vel_advected = M1.values[w_index,:]
            # double vel_advecting_int
            double [:,:] velocities = M1.values
            double [:,:] tendency_M1 = M1.tendencies
            double [:,:] flux = self.flux
            double [:,:] tendency = self.tendencies

            double [:] rho0 = Ref.rho0
            double [:] alpha0 = Ref.alpha0


        # print('MomentumAdvection: ', vel_advecting.shape, vel_advecting.size, Gr.nzg)

        for d_advected in xrange(Gr.dims):
            for k in xrange(Gr.nzg):
                flux[d_advected,k] = rho0[k]*velocities[w_index,k]*velocities[d_advected,k]
            for k in xrange(gw,Gr.nzg-gw):
                tendency[d_advected,k] = - alpha0[k]*(flux[d_advected,k+1]-flux[d_advected,k-1])*dzi2

        # for d_advected in xrange(Gr.dims):
        #     if(d_advected == 2):
        #         for k in xrange(Gr.nzg-1):
        #             vel_advecting_int = 0.5*(velocities[w_index,k]+velocities[w_index,k+1])
        #             vel_advected_int = 0.5*(velocities[d_advected,k]+velocities[d_advected,k+1])
        #             flux[d_advected,k] = rho0_half[k+1]*vel_advecting_int*vel_advected_int
        #             # print('d_advected:', d_advected, 'M1 Momentum flux:', flux[k])
        #     else:
        #         for k in xrange(Gr.nzg-1):
        #             vel_advecting_int = 0.5*(velocities[w_index,k]+velocities[w_index,k+1])
        #             vel_advected_int = 0.5*(velocities[d_advected,k]+velocities[d_advected,k+1])
        #             flux[d_advected,k] = rho0[k]*vel_advecting_int*vel_advected_int
        #             # print('d_advected:', d_advected, 'M1 Momentum flux:', flux[k])
        #     # print(d_advected, shift_advected, shift_advected+k, Gr.nzg)
        #
        #     if(d_advected == 2):
        #         for k in xrange(gw,Gr.nzg-gw):
        #             # pass
        #             tendency[d_advected,k] = - alpha0[k]*(flux[d_advected,k] - flux[d_advected,k+sm1_ed])*dzi
        #     else:
        #         for k in xrange(gw,Gr.nzg-gw):
        #             # pass
        #             tendency[d_advected,k] = - alpha0_half[k]*(flux[d_advected,k]-flux[d_advected,k+sm1_ed])*dzi

            for k in xrange(Gr.nzg):
                tendency_M1[d_advected,k] += tendency[d_advected,k]

        return



    cpdef update_M1_4th(self, Grid Gr, ReferenceState Ref, MeanVariables M1):
        print('Momentum Advection M1: update 4th !!!!!  not finished !!!!!')
        # (1) update tendencies for Mean Variables
        #       - only vertical advection
        # (1a) advection by mean velocity: 1/rho0*\partialz(rho0 <w><u_i>)
        # (1b) turbulent advection: 1/rho0*\partialz(<rho0 w'u'_i>)
        cdef:
            Py_ssize_t d_advected       # Direction of advected momentum component
            Py_ssize_t w_index = M1.name_index['w']

            Py_ssize_t k
            Py_ssize_t gw = Gr.gw
            double dzi = Gr.dzi
            double dzi2 = 0.5*Gr.dzi

            Py_ssize_t sp1_ed = 1
            Py_ssize_t sm1_ed = -sp1_ed

            # double [:] vel_advecting = M1.values[w_index,:]
            # double [:] vel_advected = M1.values[w_index,:]
            double [:,:] velocities = M1.values
            double vel_advected_ing
            double vel_advecting_int
            double [:,:] flux = self.flux
            double [:,:] tendency = self.tendencies
            double [:,:] tendency_M1 = M1.tendencies

            double [:] rho0 = Ref.rho0
            double [:] alpha0 = Ref.alpha0


        # print('MomentumAdvection: ', vel_advecting.shape, vel_advecting.size, Gr.nzg)

        for d_advected in xrange(Gr.dims):
            for k in xrange(Gr.nzg):
                flux[d_advected,k] = rho0[k]*velocities[w_index,k]*velocities[d_advected,k]
            for k in xrange(gw,Gr.nzg-gw):
                tendency[d_advected,k] = - alpha0[k]*(flux[d_advected,k+1]-flux[d_advected,k-1])*dzi2

            for k in xrange(Gr.nzg):
                tendency_M1[d_advected,k] += tendency[d_advected,k]

        return


    cpdef update_M2_2nd(self, Grid Gr, ReferenceState Ref, SecondOrderMomenta M2):
        return


    # cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return


    def plot(self, Grid Gr, TimeStepping TS, MeanVariables M1):
        if np.isnan(self.tendencies).any():
            print('!!!!! NAN in MA Tendencies M1')
            if np.isnan(self.tendencies[0,:]).any():
                print('!!!!! NAN in MA Tendencies M1, 0')
            if np.isnan(self.tendencies[1,:]).any():
                print('!!!!! NAN in MA Tendencies M1, 1')
            if np.isnan(self.tendencies[2,:]).any():
                print('!!!!! NAN in MA Tendencies M1, 2')
            if np.isnan(self.tendencies[3,:]).any():
                print('!!!!! NAN in MA Tendencies M1, 3')
        if np.isnan(self.flux).any():
            print('!!!!! NAN in MA Fluxes M1')

        print('MA', self.tendencies.shape, self.flux.shape)

        plt.figure(1,figsize=(15,7))
        # plt.plot(values[s_varshift+Gr.gw:s_varshift+Gr.nzg-Gr.gw], Gr.z)
        plt.subplot(1,5,1)
        plt.plot(M1.tendencies[3,:], Gr.z, 'x-', label='tend th')
        plt.title('M1.Tendencies')
        plt.legend(loc=3)
        plt.subplot(1,5,2)
        plt.plot(M1.tendencies[0,:], Gr.z, 'x-', label='tend u')
        plt.plot(M1.tendencies[1,:], Gr.z, 'x-', label='tend v')
        plt.plot(M1.tendencies[2,:], Gr.z, 'x-', label='tend w')
        plt.title('M1.Tendencies')
        plt.legend(loc=3)
        # plt.subplot(1,4,2)
        # plt.plot(self.tendencies[0,:], Gr.z, 'x-', label='tend u')
        # plt.plot(self.tendencies[1,:], Gr.z, 'x-', label='tend v')
        # plt.plot(self.tendencies[2,:], Gr.z, 'x-', label='tend w')
        # plt.title('self.Tendencies')
        # plt.legend(loc=3)
        plt.subplot(1,5,3)
        plt.plot(self.flux[0,:], Gr.z, 'x-', label='flux u')
        plt.plot(self.flux[1,:], Gr.z, 'x-', label='flux v')
        plt.plot(self.flux[2,:], Gr.z, 'x-', label='flux w')
        plt.title('self.Fluxes')
        plt.legend(loc=4)

        plt.subplot(1,5,4)
        plt.plot(M1.values[3,:], Gr.z, 'x-', label=' th')
        plt.title('M1.values')
        plt.legend(loc=4)
        plt.subplot(1,5,5)
        plt.plot(M1.values[0,:], Gr.z, 'x-', label=' u')
        plt.plot(M1.values[1,:], Gr.z, 'x-', label=' v')
        plt.plot(M1.values[2,:], Gr.z, 'x-', label=' w')
        plt.title('M1.values')
        plt.legend(loc=4)
        # plt.show()
        plt.savefig('./figs/MA_' + np.str(np.int(TS.t)) + '.png')
        plt.close()
        return