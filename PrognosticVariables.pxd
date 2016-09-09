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



cdef class MeanVariables:
    cdef:
        dict name_index
        list index_name
        dict units
        Py_ssize_t nv
        Py_ssize_t nv_scalars
        Py_ssize_t nv_velocities
        long [:] var_type

        double [:,:] values
        double [:,:] tendencies
    cpdef add_variable(self,name,units,var_type)
    cpdef initialize(self, Grid Gr, NetCDFIO_Stats NS)
    cpdef update(self, Grid Gr, TimeStepping TS)


cdef class SecondOrderMomenta:
    cdef:
        dict name_index
        list index_name
        dict units
        Py_ssize_t nv
        Py_ssize_t nv_scalars
        Py_ssize_t nv_velocities
        long [:] var_type

        double [:,:,:] values
        double [:,:,:] tendencies
    cpdef add_variable(self,name,units,var_type)
    # cpdef add_variables(self)
    cpdef initialize(self, Grid Gr, NetCDFIO_Stats NS)
    cpdef update(self, Grid Gr, TimeStepping TS)