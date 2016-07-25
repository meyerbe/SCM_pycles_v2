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
    # Gr.dims.npg = nlg[0] * nlg[1] * nlg[2] ( = nxg * nyg * nzg )  --> local number of pts in 3D grid incl. ghost pts


    def __init__(self,namelist):
        print('init Grid')
     #Global grid point nx, ny, nz
        self.dims = namelist['grid']['dims']

        self.nx = namelist['grid']['nx']
        self.ny = namelist['grid']['ny']
        self.nz = namelist['grid']['nz']

        #Get grid spacing from the imput file
        self.dx = namelist['grid']['dx']
        self.dy = namelist['grid']['dy']
        self.dz = namelist['grid']['dz']
        self.dxi = 1.0/self.dx
        self.dyi = 1.0/self.dy
        self.dzi = 1.0/self.dz

        #Get the dimensions of the physical domain
        self.lx = np.double(self.dx * self.nx)
        self.ly = np.double(self.dy * self.ny)
        self.lz = np.double(self.dz * self.nz)

        #The number of ghost points
        self.gw = namelist['grid']['gw']

        #Compute the global dims
        self.nxg = self.nx + 2*self.gw
        self.nyg = self.ny + 2*self.gw
        self.nzg = self.nz + 2*self.gw

        self.npd = np.max([self.nx,1])*np.max([self.ny,1])*np.max([self.nz,1])
        self.npg = self.nxg * self.nyg * self.nzg


        #Compute the coordinates
        # self.compute_global_dims()
        # self.compute_local_dims(Parallel)
        self.compute_coordinates()

        return



    def compute_coordinates(self):
        '''
        Compute the dimensional (with units) of meters coordiantes. x_half, y_half and z_half are
        the grid cell center and x,y,z are at the grid cell edges.
        :return:
        '''

        self.x_half = np.empty((self.ny+2*self.gw),dtype=np.double,order='c')
        self.x = np.empty((self.nx+2*self.gw),dtype=np.double,order='c')

        self.y_half = np.empty((self.ny+2*self.gw),dtype=np.double,order='c')
        self.y = np.empty((self.ny+2*self.gw),dtype=np.double,order='c')

        self.z_half = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')
        self.z = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')

        count = 0
        for i in xrange(-self.gw,self.nz+self.gw,1):
            self.z[count] = (i + 1) * self.dz
            self.z_half[count] = (i+0.5)*self.dz
            count += 1

        count = 0
        for i in xrange(-self.gw,self.ny+self.gw,1):
            self.y[count] = (i + 1) * self.dy
            self.y_half[count] = (i+0.5)*self.dy
            count += 1

        count = 0
        for i in xrange(-self.gw,self.nx+self.gw,1):
            self.x[count] = (i + 1) * self.dx
            self.x_half[count] = (i+0.5)*self.dx
            count += 1


        return
