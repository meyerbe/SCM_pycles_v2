#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cimport numpy as np
import numpy as np
import time


cdef class Grid:
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
    # Gr.dims.npg = nlg[0] * nlg[1] * nlg[2] ( = nxg * nyg * nzg )  --> local number of pts in 3D grid incl. ghost pts


    def __init__(self,namelist):
        #Get the grid spacing
        self.dims = namelist['grid']['dims']

        self.dz = namelist['grid']['dz']

        #Set the inverse grid spacing

        self.dzi = 1.0/self.dz

        #Get the grid dimensions and ghost points
        self.gw = namelist['grid']['gw']
        self.nz = namelist['grid']['nz']
        self.nzg = self.nz + 2 * self.gw

        self.compute_coordinates()

        return


    def compute_coordinates(self):
        '''
        Compute the dimensional (with units) of meters coordiantes. x_half, y_half and z_half are
        the grid cell center and x,y,z are at the grid cell edges.
        :return:
        '''
        self.z_half = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')
        self.z = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')

        count = 0
        for i in xrange(-self.gw,self.nz+self.gw,1):
            self.z[count] = (i + 1) * self.dz
            self.z_half[count] = (i+0.5)*self.dz
            count += 1

        return

