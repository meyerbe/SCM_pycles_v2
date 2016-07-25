#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport mpi4py.libmpi as mpi
#cimport mpi4py.mpi_c as mpi
# cimport ParallelMPI
cimport numpy as np
import numpy as np
import time


class Grid:
    '''
    A class for storing information about the LES grid.
    '''
    # i = 0,1,2
    # Gr.dims.n[i] = namelist['grid']['ni'] (e.g. n[0] = 'nx')      --> global number of pts per direction
    # Gr.dims.nl[i] = Gr.dims.n[i] // mpi_dims[i]                   --> local number of pts (per processor)

    # Gr.dims.ng[i] = Gr.dims.n[i] + 2*gw                           --> global number of pts incl. ghost pts
    # Gr.dims.nlg[i] = Gr.dims.nl[i] + 2*gw                         --> local number of pts incl ghost pts

    # Gr.dims.npd = n[0] * n[1] * n[2] ( = nx * ny * nz)            --> global number of pts in 3D grid
    # Gr.dims.npl = nl[0] * nl[1] * nl[2]                           --> local number of pts in 3D grid
    # Gr.dims.npg = nlg[0] * nlg[1] * nlg[2]                        --> local number of pts in 3D grid incl. ghost pts


    def __init__(self,namelist):
     #Global grid point nx, ny, nz
        self.dims = namelist['grid']['dims']

        self.nx = namelist['grid']['nx']
        self.ny = namelist['grid']['ny']
        self.nz = namelist['grid']['nz']

        #Get grid spacing from the imput file
        self.dx = namelist['grid']['dx']
        self.dy = namelist['grid']['dy']
        self.dz = namelist['grid']['dz']

        #Get the dimensions of the physical domain
        self.lx = np.double(self.dx * self.nx)
        self.ly = np.double(self.dy * self.ny)
        self.lz = np.double(self.dz * self.nz)


        #The number of ghost points
        self.gw = namelist['grid']['gw']

        return

