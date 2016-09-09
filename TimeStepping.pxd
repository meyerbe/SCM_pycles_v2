# cimport PrognosticVariables as PrognosticVariables
# # cimport DiagnosticVariables as DiagnosticVariables
# cimport Grid as Grid
# # cimport Restart

cdef class TimeStepping:
    cdef:
        public double dt
        public double t
        public double t_max
        public Py_ssize_t nstep
        # double [:,:] value_copies
        # double [:,:] tendency_copies
        # public Py_ssize_t ts_type


    # cpdef initialize(self, namelist, PrognosticVariables.PrognosticVariables PV)
    cpdef initialize(self, namelist)
    # cpdef update(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV)
    cpdef update(self)

    # cpdef adjust_timestep(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV)
    # cdef void compute_cfl_max(self,Grid.Grid Gr, PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV)
    # cpdef restart(self, Restart.Restart Re)

    # cdef inline double cfl_time_step(self)