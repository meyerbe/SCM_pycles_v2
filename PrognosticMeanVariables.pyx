__author__ = 'pressel'

import numpy as np
cimport numpy as np
import time
import sys

import cython
from mpi4py import MPI

class PrognosticMeanVariables:
    def __init__(self):
        self.values = None
        self.tendencies = None
        self.ndof = 0
        self.namesdof = {}
        self.units = {}
        self.bctype = []
        self.list_of_names = []
        self.bcfactor = np.empty((0,))
        self.array_dims = (None, None, None, None)
        return

    def add_variable(self, name, bctype,units=''):
        self.namesdof[name] = self.ndof
        self.units[name] = units
        self.list_of_names.append(name)
        self.bctype.append(bctype)
        if bctype=='symmetric':
            self.bcfactor = np.append(self.bcfactor,[1.0])
        elif bctype=='antisymmetric':
            self.bcfactor = np.append(self.bcfactor,[-1.0])
        else:
            print 'Warning Invalid BC Type:'
        self.ndof += 1

        return



    def initialize(self, grid):
        print('self.ndof', self.ndof, type(self.ndof))
        self.array_dims = (grid.nz, self.ndof)
        self.values = np.zeros(self.array_dims,dtype=np.double,order='C')
        self.tendencies = np.zeros(self.array_dims,dtype=np.double,order='C')

        return



    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_bcs(self, grid):
        cdef int ndof = self.ndof
        cdef int n = 0

        cdef double [:,:] values = self.values
        cdef double [:] bcfactor = self.bcfactor[:]

        cdef int i, j, k
        cdef int _gw = grid.gw
        cdef int kstart
        cdef int nzl = grid.nzl
        cdef int nxl = grid.nxl
        cdef int nyl = grid.nyl



        with nogil:
            #This processor is at the bottom of the domain so need to set bottom boundary condition
            kstart = _gw
            for k in xrange(_gw):
                for n in xrange(ndof):
                    if(bcfactor[n] == 1):
                        values[kstart-1-k,n] = values[kstart+k,n] * bcfactor[n]
                    else:
                        if(k == 0):
                            values[kstart-1-k,n] = 0.0
                        else:
                            values[kstart-1-k,n] = values[kstart+k-1,n] * bcfactor[n]

        with nogil:
            #This processor is at the top of the domain so need to set top boundary condition
            kstart = nzl - _gw
            for i in xrange(nxl):
                for j in xrange(nyl):
                    for k in xrange(_gw):
                        for n in xrange(ndof):
                            if(bcfactor[n] == 1):
                                values[kstart+k,n] = values[kstart-k-1,n] * bcfactor[n]
                            else:
                                if(k == 0):
                                    values[kstart+k,n] = 0.0
                                else:
                                    values[kstart+k,n] = values[kstart-k,n] * bcfactor[n]

        return


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_bcs_tendencies(self, grid):

        cdef int ndof = self.ndof
        cdef int n = 0
        cdef double [:,:] values = self.tendencies
        cdef double [:] bcfactor = self.bcfactor[:]


        cdef int i, j, k
        cdef int _gw = grid.gw
        cdef int kstart
        cdef int nzl = grid.nzl
        cdef int nxl = grid.nxl
        cdef int nyl = grid.nyl


        with nogil:
            #This processor is at the bottom of the domain so need to set bottom boundary condition
            kstart = _gw
            for i in xrange(nxl):
                for j in xrange(nyl):
                    for k in xrange(_gw):
                        for n in xrange(ndof):
                            if(bcfactor[n] == 1):
                                values[kstart-1-k,n] = values[kstart+k,n] * bcfactor[n]
                            else:
                                if(k == 0):
                                    values[kstart-1-k,n] = 0.0
                                else:
                                    values[kstart-1-k,n] = values[kstart+k-1,n] * bcfactor[n]


        with nogil:
            #This processor is at the top of the domain so need to set top boundary condition
            kstart = nzl - _gw
            for i in xrange(nxl):
                for j in xrange(nyl):
                    for k in xrange(_gw):
                        for n in xrange(ndof):
                            if(bcfactor[n] == 1):
                                values[kstart+k,n] = values[kstart-k-1,n] * bcfactor[n]
                            else:
                                if(k == 0):
                                    values[kstart+k,n] = 0.0
                                else:
                                    values[kstart+k,n] = values[kstart-k,n] * bcfactor[n]

        return



    def update_boundary_conditions(self,grid):
        self.update_bcs(grid)
        return

    def zero_all_tendencies(self):
        self.tendencies[:,:] = 0.0
        return

    def update_tendency_boundary_conditions(self,grid):
        self.update_bcs_tendencies(grid)
        return

    def check_nan(self,routine):
        nan_bool = np.isnan(self.tendencies)
        if nan_bool.any():
            print('Nans found!')
            print('In Routine:', routine)
            print(np.where(nan_bool))
            sys.exit()
        return


    def get_dof(self, name):
        return self.namesdof[name]

    def get_bctype(self, name):
        return self.bctype[self.namesdof[name]]