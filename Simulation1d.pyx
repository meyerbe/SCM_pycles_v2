import time
import numpy as np
cimport numpy as np
import os       # for self.outpath

cimport Grid
cimport PrognosticVariables
cimport NetCDFIO
from Thermodynamics import ThermodynamicsFactory



class Simulation1d:
    def __init__(self, namelist):
        self.Gr = Grid.Grid(namelist)
        return

    def initialize(self, namelist):
        print(type(self.Gr))
        # print('Gr.nz', self.Gr.dz)

        self.PV = PrognosticVariables.PrognosticVariables(self.Gr)

        self.Th = ThermodynamicsFactory(namelist)

        self.StatsIO = NetCDFIO.NetCDFIO_Stats()
        self.CondStatsIO = NetCDFIO.NetCDFIO_CondStats()

        uuid = str(namelist['meta']['uuid'])
        self.outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + uuid[-5:]))

        # Add new prognostic variables
        self.PV.add_variable('u', 'm/s', "sym", "velocity")
        self.PV.add_variable('v', 'm/s', "sym", "velocity")
        self.PV.add_variable('w', 'm/s', "asym", "velocity")

        # AuxillaryVariables(namelist, self.PV, self.DV, self.Pa)
        self.StatsIO.initialize(namelist, self.Gr)
        self.PV.initialize(self.Gr, self.StatsIO)

        print('Initialization completed!')
        return



    def run(self):
        print('Sim: start run')
        # cdef PrognosticVariables.PrognosticVariables PV_ = self.PV
        # cdef DiagnosticVariables.DiagnosticVariables DV_ = self.DV
        # PV_.Update_all_bcs(self.Gr, self.Pa)
        # cdef LatentHeat LH_ = self.LH
        # cdef Grid.Grid GR_ = self.Gr
        cdef int rk_step

        self.PV.update(self.Gr)

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


            # If time to ouput stats do output
            if self.CondStatsIO.last_output_time + self.CondStatsIO.frequency == self.TS.t:
            #if (1==1):
                self.Pa.root_print('Doing CondStatsIO')
                self.CondStatsIO.last_output_time = self.TS.t
                self.CondStatsIO.write_condstat_time(self.TS.t, self.Pa)

                self.CondStats.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.CondStatsIO, self.Pa)
                self.Pa.root_print('Finished Doing CondStatsIO')


            if self.VO.last_vis_time + self.VO.frequency == self.TS.t:
            #if (1==1):
                self.Pa.root_print('Dumping Visualisation File!')
                self.VO.last_vis_time = self.TS.t
                self.VO.write(self.Gr, self.Ref, self.PV, self.DV, self.Pa)


            if self.Restart.last_restart_time + self.Restart.frequency == self.TS.t:
                self.Pa.root_print('Dumping Restart Files!')
                self.Restart.last_restart_time = self.TS.t
                self.Restart.restart_data['last_stats_output'] = self.StatsIO.last_output_time
                self.Restart.restart_data['last_condstats_output'] = self.CondStatsIO.last_output_time
                self.Restart.restart_data['last_vis_time'] = self.VO.last_vis_time
                self.Gr.restart(self.Restart)
                self.Sur.restart(self.Restart)
                self.Ref.restart(self.Gr, self.Restart)
                self.PV.restart(self.Gr, self.Restart)
                self.TS.restart(self.Restart)

                self.Restart.write(self.Pa)
                self.Pa.root_print('Finished Dumping Restart Files!')

        return







