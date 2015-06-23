__author__ = 'pressel'

import numpy as np
cimport numpy as np
import sys

class Grid:
    def __init__(self ,nml):

        #Global grid point nx, ny, nz
        self.nx = nml.grid['nx']
        self.ny = nml.grid['ny']
        self.nz = nml.grid['nz']

        #Get grid spacing from the imput file
        self.dx = nml.grid['dx']
        self.dy = nml.grid['dy']
        self.dz = nml.grid['dz']

        #Get the dimensions of the physical domain
        self.lx = np.double(self.dx * self.nx)
        self.ly = np.double(self.dy * self.ny)
        self.lz = np.double(self.dz * self.nz)


        #The number of ghost points
        self.gw = nml.grid['gw']

        self.nxg = self.nx + 2 * self.gw
        self.nyg = self.ny + 2 * self.gw
        self.nzg = self.nz + 2 * self.gw
