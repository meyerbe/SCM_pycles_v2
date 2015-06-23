__author__ = 'pressel'

import numpy as np
cimport numpy as np
import time
import sys

import cython
from mpi4py import MPI

class PrognosticCumulants:
# !!! define on what grid ???
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

    def add_variable(self, name, d1, d2, bctype,units=''):
        self.namesdof[name] = [d1, d2]
        self.units[name] = units
        self.list_of_names.append(name)
        #self.ndof += 1

        self.bctype.append(bctype)
        if bctype=='symmetric':
            self.bcfactor = np.append(self.bcfactor,[1.0])
        elif bctype=='antisymmetric':
            self.bcfactor = np.append(self.bcfactor,[-1.0])
        else:
            print 'Warning Invalid BC Type:'

        return



    def initialize(self, grid):
        print('cumulants: self.ndof', self.ndof, type(self.ndof))

        self.array_dims = (grid.nz, grid.nz, self.ndof, self.ndof)
        self.values = np.zeros(self.array_dims,dtype=np.double,order='C')
        self.tendencies = np.zeros(self.array_dims,dtype=np.double,order='C')
        print('c', self.values.shape)
        return


    def zero_all_tendencies(self):
        self.tendencies[:,:,:] = 0.0
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