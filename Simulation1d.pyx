
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
from SGS import SGSFactory
cimport MomentumDiffusion
cimport ScalarDiffusion
cimport NetCDFIO
from Thermodynamics import ThermodynamicsFactory
from TurbulenceScheme import TurbulenceFactory
# # cimport TurbulenceScheme



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
        self.Turb = TurbulenceFactory(namelist)

        self.SGS = SGSFactory(namelist)
        self.MD = MomentumDiffusion.MomentumDiffusion()
        self.SD = ScalarDiffusion.ScalarDiffusion(namelist)

        self.StatsIO = NetCDFIO.NetCDFIO_Stats()
        return

    def initialize(self, namelist):

        uuid = str(namelist['meta']['uuid'])
        self.outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + uuid[-5:]))
        self.StatsIO.initialize(namelist, self.Gr)

        self.TS.initialize(namelist)

        # Add new prognostic variables
        self.PV.add_variable('phi', 'm/s', "velocity")      # self.PV.add_variable('phi', 'm/s', "sym", "velocity")
        # cdef:
        #     PV_ = self.PV
        # print(PV_.nv)     #not accessible (also not as # print(self.PV.nv))
        self.M1.add_variable('u', 'm/s', "velocity")
        self.M1.add_variable('v', 'm/s', "velocity")
        self.M1.add_variable('w', 'm/s', "velocity")

        # AuxillaryVariables(namelist, self.PV, self.DV, self.Pa)
        self.Th.initialize(self.Gr, self.M1, self.M2)        # adding prognostic thermodynamic variables
        self.PV.initialize(self.Gr, self.StatsIO)
        self.M1.initialize(self.Gr, self.StatsIO)
        self.M2.initialize(self.Gr, self.M1, self.StatsIO)

        self.Init.initialize_reference(self.Gr, self.Ref, self.StatsIO)
        self.Init.initialize_profiles(self.Gr, self.Ref, self.M1, self.M2, self.StatsIO)

        self.MA.initialize(self.Gr, self.M1)
        self.SA.initialize(self.Gr, self.M1)
        self.Turb.initialize(self.Gr, self.M1)
        self.SGS.initialize(self.Gr, self.M1, self.M2)
        self.MD.initialize(self.Gr, self.M1)
        self.SD.initialize(self.Gr, self.M1)

        print('Initialization completed!')
        # self.plot()
        return



    def run(self):
        print('Sim: start run')
        print(self.TS.t, self.TS.t_max)

        while(self.TS.t < self.TS.t_max):
            print('time:', self.TS.t)
            # pass
            # (0) update auxiliary fields
            self.SGS.update(self.Gr)       # --> compute diffusivity / viscosity for M1 and M2 (being the same at the moment)
            # self.M1.plot('beginning of timestep', self.Gr, self.TS)
            # (1) update mean field (M1) tendencies
            self.Th.update()
            self.MA.update_M1_2nd(self.Gr, self.Ref, self.M1)       # self.MA.update(self.Gr, self.Ref, self.M1)
            self.SA.update_M1_2nd(self.Gr, self.Ref, self.M1)       # self.SA.update(self.Gr, self.Ref, self.M1)

            self.MD.update(self.Gr, self.Ref, self.M1, self.SGS)
            self.SD.update(self.Gr, self.Ref, self.M1, self.SGS)
            # self.M1.plot('after SD update', self.Gr, self.TS)

            self.Turb.update_M1(self.Gr, self.Ref, self.M1, self.M2)                         # --> add turbulent flux divergence to mean field tendencies: dz<w'phi'>
            # ??? surface fluxes ??? (--> in SGS or MD/SD scheme?)
            # ??? update boundary conditions ???
            # ??? pressure solver ???

            # self.M1.plot('without tendency update', self.Gr, self.TS)


            # (2) update second order momenta (M2) tendencies
            # self.MA.update                        # --> self.MA.update_M2(): advection of M2
            # self.SA.update                        # --> self.SA.update_M2(): advection of M2
            # self.MD.update()
            # self.SD.update()
            # self.Turb.update_M2()                 # update higher order terms in M2 tendencies
            # print('Sim: Turb update')
            self.Turb.update(self.Gr, self.Ref, self.M1, self.M2)
            #     # Turb.advect_M2_local(Gr, M1, M2)
            # # ??? update boundary conditions???
            # # ??? pressure correlations ???
            # # ??? surface fluxes ??? (--> in SGS or MD/SD scheme?)


            self.M1.update(self.Gr, self.TS)        # --> updating values by adding tendencies
            self.M2.update(self.Gr, self.TS)        # --> updating values by adding tendencies
            self.TS.update()

        return





    # def plot(self):
    #
    #     plt.figure(1)
    #     plt.plot()
    #     plt.title(var + ', ' + message)
    #     # plt.show()
    #     plt.savefig(self.outpath + '/' + var + '_' + message + '.png')
    #     plt.close()





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







