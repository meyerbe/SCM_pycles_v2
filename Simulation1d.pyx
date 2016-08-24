
# ??? Surface
# ??? Boundary Conditions
# Statistical Output: make different grouops of variables for Mean Var and Second Order Momenta


import time
import numpy as np
cimport numpy as np
import os       # for self.outpath

from Grid cimport Grid
cimport TimeStepping
cimport ReferenceState
from Initialization import InitializationFactory
cimport PrognosticVariables
cimport MomentumAdvection
cimport ScalarAdvection
cimport MomentumDiffusion
cimport ScalarDiffusion
cimport NetCDFIO
from Thermodynamics import ThermodynamicsFactory



class Simulation1d:
    def __init__(self, namelist):
        self.Gr = Grid(namelist)
        self.TS = TimeStepping.TimeStepping()
        self.Ref = ReferenceState.ReferenceState(self.Gr)
        self.Init = InitializationFactory(namelist)

        self.PV = PrognosticVariables.PrognosticVariables(self.Gr)
        self.M1 = PrognosticVariables.MeanVariables(self.Gr)
        self.M2 = PrognosticVariables.SecondOrderMomenta(self.Gr)

        self.Th = ThermodynamicsFactory(namelist)

        self.MA = MomentumAdvection.MomentumAdvection(namelist)
        self.SA = ScalarAdvection.ScalarAdvection(namelist)

        self.MD = MomentumDiffusion.MomentumDiffusion()
        self.SD = ScalarDiffusion.ScalarDiffusion(namelist)

        self.StatsIO = NetCDFIO.NetCDFIO_Stats()
        return

    def initialize(self, namelist):
        # print(type(self.Gr))
        # print('Gr.nz', self.Gr.dz)

        uuid = str(namelist['meta']['uuid'])
        self.outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + uuid[-5:]))
        self.StatsIO.initialize(namelist, self.Gr)
            # cpdef initialize(self, dict namelist, Grid.Grid Gr):


        self.TS.initialize(namelist)
        # self.Ref.initialize(self.Gr, self.Th, self.StatsIO)            # Reference State
        self.Init.initialize_reference(self.Gr, self.Ref, self.StatsIO)
        self.Init.initialize_profiles(self.Gr, self.Ref, self.M1, self.M2)

        # Add new prognostic variables
        self.PV.add_variable('phi', 'm/s', "sym", "velocity")
        self.M1.add_variable('w', 'm/s', "velocity")
        self.M2.add_variable('ww', '(m/s)^2', "velocity")

        # AuxillaryVariables(namelist, self.PV, self.DV, self.Pa)
        # self.StatsIO.initialize(namelist, self.Gr)
        self.PV.initialize(self.Gr, self.StatsIO)
        self.M1.initialize(self.Gr, self.StatsIO)
        self.M2.initialize(self.Gr, self.StatsIO)

        self.Th.initialize(self.Gr, self.PV)
        self.MA.initialize()
        self.SA.initialize()
        self.MD.initialize(self.Gr)
        self.SD.initialize(self.Gr)

        print('Initialization completed!')
        return



    def run(self):
        print('Sim: start run')

        while(self.TS.t < self.TS.t_max):

            # pass

            # update PV tendencies
            self.Th.update()
            self.MA.update()
            self.SA.update()
            self.MD.update()
            self.SD.update()
            #
            # # update PV tendencies
            # self.PV.update(self.Gr, self.TS) # !!! causes error !!!
            # # print('PV.update')
            self.M1.update(self.Gr, self.TS)
            # # print('M1.update')
            self.M2.update(self.Gr, self.TS)
            # # print('M2.update')
            #
            # # ??? update boundary conditions???
            self.TS.update()

        return










    def io(self):
        print('calling io')
        cdef:
            double stats_dt = 0.0
            double condstats_dt = 0.0
            double restart_dt = 0.0
            double vis_dt = 0.0
            double min_dt = 0.0

        if self.TS.t > 0 and self.TS.rk_step == self.TS.n_rk_steps - 1:
            print('doing io: ' + str(self.TS.t) + ', ' + str(self.TS.rk_step))
            # Adjust time step for output if necessary
            stats_dt = self.StatsIO.last_output_time + self.StatsIO.frequency - self.TS.t
            condstats_dt = self.CondStatsIO.last_output_time + self.CondStatsIO.frequency - self.TS.t
            restart_dt = self.Restart.last_restart_time + self.Restart.frequency - self.TS.t
            vis_dt = self.VO.last_vis_time + self.VO.frequency - self.TS.t


            # dts = np.array([fields_dt, stats_dt, condstats_dt, restart_dt, vis_dt,
            #                 self.TS.dt, self.TS.dt_max, self.VO.frequency, self.Restart.frequency,
            #                 self.StatsIO.frequency, self.CondStatsIO.frequency, self.FieldsIO.frequency])
            dts = np.array([stats_dt, condstats_dt, restart_dt, vis_dt,
                            self.TS.dt, self.TS.dt_max, self.VO.frequency, self.Restart.frequency,
                            self.StatsIO.frequency, self.CondStatsIO.frequency])



            self.TS.dt = np.amin(dts[dts > 0.0])
            # If time to ouput stats do output
            # self.Pa.root_print('StatsIO freq: ' + str(self.StatsIO.frequency) + ', ' + str(self.StatsIO.last_output_time) + ', ' + str(self.TS.t))
            if self.StatsIO.last_output_time + self.StatsIO.frequency == self.TS.t:
            #if self.StatsIO.last_output_time + self.StatsIO.frequency == self.TS.t or self.StatsIO.last_output_time + self.StatsIO.frequency + 10.0 == self.TS.t:
            #if (1==1):
                self.Pa.root_print('Doing StatsIO')
                self.StatsIO.last_output_time = self.TS.t
                self.StatsIO.open_files(self.Pa)
                self.StatsIO.write_simulation_time(self.TS.t, self.Pa)
                self.Micro.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa) # do Micro.stats_io prior to DV.stats_io to get sedimentation velocity only in output
                self.PV.stats_io(self.Gr, self.Ref, self.StatsIO, self.Pa)

                self.DV.stats_io(self.Gr, self.StatsIO, self.Pa)
                self.Fo.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)
                self.Th.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.StatsIO, self.Pa)

                self.Sur.stats_io(self.Gr, self.StatsIO, self.Pa)
                self.SGS.stats_io(self.Gr,self.DV,self.PV,self.Ke,self.StatsIO,self.Pa)
                self.SA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
                self.MA.stats_io(self.Gr, self.PV, self.StatsIO, self.Pa)
                self.SD.stats_io(self.Gr, self.Ref,self.PV, self.DV, self.StatsIO, self.Pa)
                self.MD.stats_io(self.Gr, self.PV, self.DV, self.Ke, self.StatsIO, self.Pa)
                self.Ke.stats_io(self.Gr,self.Ref,self.PV,self.StatsIO,self.Pa)
                self.Tr.stats_io( self.Gr, self.StatsIO, self.Pa)
                self.Ra.stats_io(self.Gr, self.DV, self.StatsIO, self.Pa)
                self.Budg.stats_io(self.Sur, self.StatsIO, self.Pa)
                self.Aux.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.MA, self.MD, self.StatsIO, self.Pa)
                self.StatsIO.close_files(self.Pa)
                self.Pa.root_print('Finished Doing StatsIO')

        return







