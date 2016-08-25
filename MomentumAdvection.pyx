#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

from Grid cimport Grid
from PrognosticVariables cimport MeanVariables
from ReferenceState cimport ReferenceState
# from NetCDFIO cimport NetCDFIO_Stats

import numpy as np
cimport numpy as np
import sys

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

    # cpdef initialize(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef initialize(self, Grid Gr, MeanVariables M1):
        # initialize scheme for Mean Variables & Second Order Momenta
        self.flux = np.zeros((M1.nv_velocities*Gr.nzg),dtype=np.double,order='c')

        return

    # cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV):
    cpdef update(self, Grid Gr, ReferenceState Ref, MeanVariables M1):
        # (1) update tendencies for Mean Variables
        #       - only vertical advection
        if self.order == 2:
            self.update_M1_2nd(Gr, Ref, M1)


        # (2) update tendencies for 2nd Order Momenta
        return





    # cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Rs, PrognosticVariables.PrognosticVariables PV):
    cpdef update_M1_2nd(self, Grid Gr, ReferenceState Ref, MeanVariables M1):
        print('update 2nd')
        # (1) update tendencies for Mean Variables
        #       - only vertical advection
        # (1a) advection by mean velocity: 1/rho0*\partialz(rho0 <w><u_i>)
        # (1b) turbulent advection: 1/rho0*\partialz(<rho0 w'u'_i>)
        cdef:
            Py_ssize_t d_advected       # Direction of advected momentum component
            Py_ssize_t shift_advected
            Py_ssize_t w_varshift = M1.get_varshift(Gr,'w')

            Py_ssize_t k
            Py_ssize_t kmin = 1
            Py_ssize_t kmax = Gr.nzg

            # Py_ssize_t stencil[3] = {istride,jstride,1}
            Py_ssize_t sp1_ed = 1
            Py_ssize_t sp2_ed = 2 * sp1_ed
            Py_ssize_t sm1_ed = -sp1_ed

            # double [:] vel_advecting = M1.values[w_varshift:(w_varshift+kmax)]
            # double [:] vel_advected = M1.values[w_varshift:(w_varshift+kmax)]
            double [:] velocities = M1.values
            double vel_advected_ing
            double vel_advecting_int
            double [:] flux = self.flux

            double [:] rho0 = Ref.rho0
            double [:] rho0_half = Ref.rho0_half

        # print('MomentumAdvection: ', vel_advecting.shape, vel_advecting.size, Gr.nzg)

        for d_advected in xrange(Gr.dims):
            shift_advected = M1.velocity_directions[d_advected] * Gr.nzg
            if(d_advected == 2):
                for k in xrange(kmax):
                    # vel_advecting_int = 0.5*(vel_advecting[k]+vel_advecting[k+1])
                    vel_advecting_int = 0.5*(velocities[w_varshift+k]+velocities[w_varshift+k+1])
                    vel_advected_int = 0.5*(velocities[shift_advected+k]+velocities[shift_advected+k+1])
                    flux[k] = rho0_half[k+1]*vel_advecting_int*vel_advected_int
            else:
                for k in xrange(kmax):
                    vel_advecting_int = 0.5*(velocities[w_varshift+k]+velocities[w_varshift+k+1])
                    vel_advected_int = 0.5*(velocities[shift_advected+k]+velocities[shift_advected+k+1])
                    flux[k] = rho0[k]*vel_advecting_int*vel_advected_int

        return


    # cpdef stats_io(self, Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, NetCDFIO_Stats NS):
    cpdef stats_io(self):

        return
