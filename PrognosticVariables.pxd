from NetCDFIO cimport NetCDFIO_Stats
from Grid cimport Grid
from TimeStepping cimport TimeStepping

cdef class PrognosticVariables:
    cdef:
        dict name_index
        list index_name
        dict units
        Py_ssize_t nv
        Py_ssize_t nv_scalars
        Py_ssize_t nv_velocities
        long [:] var_type
    cdef double [:] values
    cdef double [:] tendencies

    cpdef add_variable(self,name,units,var_type)
    cpdef initialize(self, Grid Gr, NetCDFIO_Stats NS)
    cpdef update(self, Grid Gr, TimeStepping TS)
    cdef inline Py_ssize_t get_nv(self, str variable_name):
        return self.name_index[variable_name]



cdef class MeanVariables:
    cdef:
        dict name_index
        list index_name
        dict units
        Py_ssize_t nv
        Py_ssize_t nv_scalars
        Py_ssize_t nv_velocities
        long [:] var_type
        long [:] velocity_directions

        double [:,:] values
        double [:,:] tendencies
    cpdef add_variable(self,name,units,var_type)
    cpdef initialize(self, Grid Gr, NetCDFIO_Stats NS)
    cpdef update(self, Grid Gr, TimeStepping TS)
    cpdef plot(self, str message, Grid Gr, TimeStepping TS)
    cpdef plot_tendencies(self, str message, Grid Gr, TimeStepping TS)
    cdef inline Py_ssize_t get_nv(self, str variable_name):
        return self.name_index[variable_name]


cdef class SecondOrderMomenta:
    cdef:
        dict name_index
        list index_name
        dict var_index
        dict units
        Py_ssize_t nv
        Py_ssize_t nv_scalars
        Py_ssize_t nv_velocities
        long [:] var_type

        double [:,:,:] values
        double [:,:,:] tendencies
    cpdef add_variable(self,name,units,var_type,n,m)
    # cpdef add_variables(self)
    cpdef initialize(self, Grid Gr, MeanVariables M1, NetCDFIO_Stats NS)
    cpdef update(self, Grid Gr, TimeStepping TS)
    cpdef plot(self, str message, Grid Gr, TimeStepping TS)
    cpdef plot_tendencies(self, str message, Grid Gr, TimeStepping TS)
    cdef inline Py_ssize_t get_nv(self, str variable_name):
        return self.name_index[variable_name]