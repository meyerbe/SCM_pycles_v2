__author__ = 'pressel'

import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmax
import cython
from cython.parallel import prange

from mpi4py import MPI
import pylab as plt
import time
import sys

class TimeStepping2ndSSP:

    def __init__(self,nml):
        try:
            self.target_dt = nml.time['dt']
        except:
            self.target_dt = 0.1
        try:
            self.timemax = nml.time['timemax']
        except:
            self.timemax = 1.0
        try:
            self.cfl_target = nml.time['cfl_target']
        except:
            print('CFL target defaulted')
            self.cfl_target = 0.7

        try:
            self.dt_fixed = nml.time['dt_fixed']
        except:
            self.dt_fixed = False

        self.time = 0.0

        self.rk_step = 0
        self.num_rk_step = 2

        self.cfl_max = 0.0

        self.dt = self.target_dt




    def initialize(self,scalars,velocities,grid):
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update(self,scalars,velocities,grid,io,comm):

        #velocities.check_nan('Time Stepping')
        #scalars.check_nan('Time Stepping')

        cdef double [:,:,:,:] scalars_tendencies = scalars.tendencies[:,:,:,:]
        cdef double [:,:,:,:] scalars_values = scalars.values[:,:,:,:]
        cdef int scalars_num = scalars.ndof

        cdef double [:,:,:,:] velocities_tendencies = velocities.tendencies[:,:,:,:]
        cdef double [:,:,:,:] velocities_values = velocities.values[:,:,:,:]
        cdef int velocities_num = velocities.ndof

        cdef int nxl = grid.nxl
        cdef int nyl = grid.nyl
        cdef int nzl = grid.nzl

        cdef double dt = self.dt

        cdef int i,j,k, n

        cdef double [:,:,:,:] phi_n_scalars = self.phi_n_scalars
        cdef double [:,:,:,:] phi_n_velocities = self.phi_n_velocities

        if self.rk_step == 0:
            with nogil:
                for i in prange(nxl,schedule='static'):
                    for j in xrange(nyl):
                        for k in xrange(nzl):
                            for n in xrange(velocities_num):
                                phi_n_velocities[i,j,k,n] = velocities_values[i,j,k,n]
                                velocities_values[i,j,k,n] = velocities_values[i,j,k,n] +  velocities_tendencies[i,j,k,n] * dt
                                velocities_tendencies[i,j,k,n] = 0.0

                            for n in xrange(scalars_num):
                                phi_n_scalars[i,j,k,n] = scalars_values[i,j,k,n]
                                scalars_values[i,j,k,n] = scalars_values[i,j,k,n] +  scalars_tendencies[i,j,k,n] * dt
                                scalars_tendencies[i,j,k,n] = 0.0

        else:
            with nogil:
                for i in prange(nxl,schedule='static'):
                    for j in xrange(nyl):
                        for k in xrange(nzl):
                            for n in xrange(velocities_num):
                                velocities_values[i,j,k,n] = 0.5 *(phi_n_velocities[i,j,k,n] +
                                                               (velocities_values[i,j,k,n]  + velocities_tendencies[i,j,k,n] * dt ))
                                velocities_tendencies[i,j,k,n] = 0.0



                            for n in xrange(scalars_num):
                                scalars_values[i,j,k,n] = 0.5 *(phi_n_scalars[i,j,k,n] +
                                                                 (scalars_values[i,j,k,n] + scalars_tendencies[i,j,k,n] * dt ))
                                scalars_tendencies[i,j,k,n] = 0.0


            self.time += dt




