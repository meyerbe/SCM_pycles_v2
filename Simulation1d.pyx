
# ??? Surface
# ??? Boundary Conditions
# Statistical Output: make different grouops of variables for Mean Var and Second Order Momenta


import time
import numpy as np
cimport numpy as np
import pylab as plt
import os       # for self.outpath

from Grid cimport Grid
cimport TimeStepping
cimport ReferenceState
from Initialization import InitializationFactory
cimport PrognosticVariables
cimport MomentumAdvection
cimport ScalarAdvection
from SGS import SGSFactory
cimport Diffusion
cimport NetCDFIO
from Thermodynamics import ThermodynamicsFactory
from TurbulenceScheme import TurbulenceFactory
cimport Damping
# # cimport TurbulenceScheme



class Simulation1d:
    def __init__(self, namelist):
        self.Gr = Grid(namelist)
        self.TS = TimeStepping.TimeStepping()
        self.Ref = ReferenceState.ReferenceState(self.Gr)
        self.Damp = Damping.Damping(namelist)
        self.Init = InitializationFactory(namelist)

        self.PV = PrognosticVariables.PrognosticVariables(self.Gr)
        self.M1 = PrognosticVariables.MeanVariables(self.Gr)
        self.M2 = PrognosticVariables.SecondOrderMomenta(self.Gr)

        self.Th = ThermodynamicsFactory(namelist)

        self.MA = MomentumAdvection.MomentumAdvection(namelist)
        self.SA = ScalarAdvection.ScalarAdvection(namelist)
        self.Turb = TurbulenceFactory(namelist)

        self.SGS = SGSFactory(namelist)
        self.Diff = Diffusion.Diffusion()

        self.StatsIO = NetCDFIO.NetCDFIO_Stats()
        return

    def initialize(self, namelist):

        uuid = str(namelist['meta']['uuid'])
        self.outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + uuid[-5:]))
        self.StatsIO.initialize(namelist, self.Gr)

        self.TS.initialize(namelist)

        # Add new prognostic variables
        self.PV.add_variable('phi', 'm/s', "velocity")      # self.PV.add_variable('phi', 'm/s', "sym", "velocity")
        self.M1.add_variable('u', 'm/s', "sym", "velocity")
        self.M1.add_variable('v', 'm/s', "sym", "velocity")
        self.M1.add_variable('w', 'm/s', "asym", "velocity")
        # self.M1.add_variable('w', 'm/s', "sym", "velocity")

        # AuxillaryVariables(namelist, self.PV, self.DV, self.Pa)
        self.Th.initialize(self.Gr, self.M1, self.M2)        # adding prognostic thermodynamic variables
        self.PV.initialize(self.Gr, self.StatsIO)
        self.M1.initialize(self.Gr, self.StatsIO)
        self.M2.initialize(self.Gr, self.M1, self.StatsIO)
        # self.M2.plot_tendencies('1', self.Gr, self.TS)

        self.Init.initialize_reference(self.Gr, self.Ref, self.StatsIO)
        self.Init.initialize_profiles(self.Gr, self.Ref, self.TS, self.M1, self.M2, self.StatsIO)
        # self.M2.plot_tendencies('2', self.Gr, self.TS)

        self.MA.initialize(self.Gr, self.M1)
        self.SA.initialize(self.Gr, self.M1)
        self.Turb.initialize(self.Gr, self.M1)

        self.SGS.initialize(self.Gr, self.M1, self.M2)
        self.Diff.initialize(self.Gr, self.M1)
        # self.M2.plot_tendencies('3', self.Gr, self.TS)

        print('Initialization completed!')
        self.M1.plot_tendencies('init', self.Gr, self.TS)
        self.M2.plot_tendencies('init', self.Gr, self.TS)
        self.M1.plot('init', self.Gr, self.TS)
        self.M2.plot('init', self.Gr, self.TS)

        self.StatsIO.update(self.Gr, self.TS, self.M1, self.M2)
        return



    def run(self):
        print('Sim: start run')
        print(self.TS.t, self.TS.t_max)

        while(self.TS.t < self.TS.t_max):
            print('time:', self.TS.t)
            self.M1.plot('1_start', self.Gr, self.TS)
            self.M1.plot_tendencies('1_start', self.Gr, self.TS)
            self.M2.plot('1_start', self.Gr, self.TS)
            self.M2.plot_tendencies('1_start', self.Gr, self.TS)

            # (0) update auxiliary fields
            self.SGS.update(self.Gr)       # --> compute diffusivity / viscosity for M1 and M2 (being the same at the moment)
            ## self.SGS.plot(self.Gr, self.TS)

            # (1) update mean field (M1) tendencies
            self.Th.update()        # --> does nothing, since mean buoyancy approximated to be zero (do buoyancy update; add to w-tend (so far no coupling btw. thermodynamics and dynamics))
            # self.MA.update_M1_2nd(self.Gr, self.Ref, self.M1)       # self.MA.update(self.Gr, self.Ref, self.M1)
            # self.SA.update_M1_2nd(self.Gr, self.Ref, self.M1)       # self.SA.update(self.Gr, self.Ref, self.M1)
            ## self.MA.plot(self.Gr, self.TS, self.M1)
            ## self.SA.plot(self.Gr, self.TS, self.M1)

            self.Diff.update_M1(self.Gr, self.Ref, self.M1, self.SGS)
            self.Diff.plot(self.Gr, self.TS, self.M1)
            self.M1.plot_tendencies('2_after_Diffusion', self.Gr, self.TS)

            self.Turb.update_M1(self.Gr, self.Ref, self.TS, self.M1, self.M2)                         # --> add turbulent flux divergence to mean field tendencies: dz<w'phi'>
            self.M1.plot_tendencies('3_after_Turb', self.Gr, self.TS)

            # (2) update second order momenta (M2) tendencies
            # self.Turb.update_M2(self.Gr, self.Ref, self.TS, self.M1, self.M2)
            # self.Turb.plot('Turb', self.Gr, self.TS, self.M1, self.M2)
                    # # ??? update boundary conditions???
                    # # ??? pressure correlations ???
                    # # ??? surface fluxes ??? (--> in SGS or MD/SD scheme?)
            # self.M2.plot_tendencies('2_after_Turb',self.Gr,self.TS)

            self.M2.plot_tendencies('end', self.Gr, self.TS)
            self.M1.plot_tendencies('end', self.Gr, self.TS)

            self.M1.update(self.Gr, self.TS)        # --> updating values by adding tendencies
            self.M2.update(self.Gr, self.TS)        # --> updating values by adding tendencies

            self.M1.plot_tendencies('control', self.Gr, self.TS)
            self.M2.plot_tendencies('control', self.Gr, self.TS)
            self.TS.update()
            self.M1.plot('before_bcs', self.Gr, self.TS)
            self.M2.plot('before_bcs', self.Gr, self.TS)
            self.M1.update_boundary_conditions(self.Gr)
            self.M1.update_boundary_conditions_tendencies(self.Gr)
            self.M1.plot('end', self.Gr, self.TS)
            self.M1.plot_tendencies('end', self.Gr, self.TS)
            self.M2.plot('end', self.Gr, self.TS)

            # (3) IO
            print('statsio', self.StatsIO.last_output_time, self.StatsIO.frequency, self.TS.t)
            if self.StatsIO.last_output_time + self.StatsIO.frequency == self.TS.t:
                print('do StatsIO.update')
                self.StatsIO.update(self.Gr, self.TS, self.M1, self.M2)

        return



    def plot_M1(self,message,pv_name):
        plt.figure(1,figsize=(12,6))
        var_list = ['w','th']
        i = 1
        for var_name in var_list:
            var_val = self.M1.get_variable_array(var_name,self.Gr)
            var_tend = self.M1.get_tendency_array(var_name,self.Gr)
            plt.subplot(2,2,np.int(i))
            plt.plot(var_val)
            plt.subplot(2,2,np.int(i)+2)
            plt.plot(var_val)
            # plt.plot(var,self.Gr.z)
            plt.title(var_name + ', ' + message + ', time: ' + np.str(np.int(self.TS.t)))
            plt.plot(var_tend)
            plt.title(var_name + '_tend , ' + message + ', time: ' + np.str(np.int(self.TS.t)))
            i += 1
        # plt.show()
        plt.savefig('figs/M1_all_' + message + '_' + np.str(np.int(self.TS.t)) + '.png')
        # plt.savefig(self.outpath + '/' + var_name + '_' + message + '_' + np.str(self.TS.t) + '.png')
        plt.close()
        return


    def plot(self,message,var_name,array_type,pv_name):
        # if pv_name == 'M1':
        #     cdef:
        #         Py_ssize_t var_index = 1
        #         double [:] var =

        if pv_name == 'M1':
            if array_type == 'value':
                var = self.M1.get_variable_array(var_name,self.Gr)
            elif array_type == 'tendency':
                var = self.M1.get_tendency_array(var_name,self.Gr)
                var_name = var_name + '_tend'

        plt.figure(1,figsize=(12,6))
        # plt.plot(var,self.Gr.z)
        plt.plot(var)
        plt.title(var_name + ', ' + message + ', time: ' + np.str(self.TS.t))
        # plt.show()
        plt.savefig('figs/' + var_name + '_' + message + '_' + np.str(self.TS.t) + '.png')
        # plt.savefig(self.outpath + '/' + var_name + '_' + message + '_' + np.str(self.TS.t) + '.png')
        plt.close()
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
                self.Diff.stats_io()
                self.Ke.stats_io(self.Gr,self.Ref,self.PV,self.StatsIO,self.Pa)
                self.Tr.stats_io( self.Gr, self.StatsIO, self.Pa)
                self.Ra.stats_io(self.Gr, self.DV, self.StatsIO, self.Pa)
                self.Budg.stats_io(self.Sur, self.StatsIO, self.Pa)
                # self.Aux.stats_io(self.Gr, self.Ref, self.PV, self.DV, self.MA, self.MD, self.StatsIO, self.Pa)
                self.StatsIO.close_files(self.Pa)
                self.Pa.root_print('Finished Doing StatsIO')

        return







