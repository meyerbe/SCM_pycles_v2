__author__ = 'bettinameyer'


import time
import numpy as np
import scipy.stats as stats
import matplotlib.cm as cm
import pylab as plt
from matplotlib import colors, ticker
from matplotlib.colors import LogNorm

from Namelist import Namelist
from Grid import Grid
from PrognosticMeanVariables import PrognosticMeanVariables
from PrognosticCumulants import PrognosticCumulants
from TimeStepping2ndSSP import TimeStepping2ndSSP
from Advection2nd_MeanMomentum import MomentumAdvection2nd
from Advection2nd_Cumulants import CumulantAdvection2nd


class Simulation:

    def __init__(self):
        self.nml =                  Namelist()

        self.velocities_mean =      PrognosticMeanVariables()
        self.scalars_mean =         PrognosticMeanVariables()
        self.pressure_mean =        PrognosticMeanVariables()

        self.cumulants =            PrognosticCumulants()

        self.grid =                 Grid(self.nml)           # do only z-Grid

        self.timestepping =         TimeStepping2ndSSP(self.nml)
        self.mean_momentumadvection =    MomentumAdvection2nd(self.grid)
        self.cumulant_advection =    CumulantAdvection2nd()


    def initialize(self):
        print('Sim.initialize()')
        # Add Mean Fields as Prognostic Variables
        self.velocities_mean.add_variable('u','symmetric',units='m s^-1')
        self.velocities_mean.add_variable('v','symmetric',units='m s^-1')
        self.velocities_mean.add_variable('w','antisymmetric',units='m s^-1')

        self.scalars_mean.add_variable('potential_temperature','symmetric',units='K')
        #self.scalars_mean.add_variable('specific_entropy','symmetric',units='K')
        self.pressure_mean.add_variable('p','symmetric',units='Pa')

        vdof = self.velocities_mean.ndof
        sdof = self.scalars_mean.ndof

        list_of_names = self.velocities_mean.list_of_names + self.scalars_mean.list_of_names + self.pressure_mean.list_of_names
        #print(list_of_names)
        #print(self.scalars_mean.list_of_names, self.scalars_mean.get_dof('potential_temperature'), self.scalars_mean.units['potential_temperature'])

        for name1 in list_of_names:
            self.cumulants.ndof += 1
            for name2 in list_of_names:
                print(name1, name2)
                try:
                    d1 = self.velocities_mean.get_dof(name1)
                    unit1 = self.velocities_mean.units[name1]
                except:
                    try:
                        d1 = vdof + self.scalars_mean.get_dof(name1)
                        unit1 = self.scalars_mean.units[name1]
                    except:
                        d1 = vdof + sdof + self.pressure_mean.get_dof(name1)
                        unit1 = self.pressure_mean.units[name1]

                try:
                    d2 = self.velocities_mean.get_dof(name2)
                    unit2 = self.velocities_mean.units[name2]
                except:
                    try:
                        d2 = vdof + self.scalars_mean.get_dof(name2)
                        unit2 = self.scalars_mean.units[name2]
                    except:
                        d2 = vdof + sdof + self.pressure_mean.get_dof(name2)
                        unit2 = self.pressure_mean.units[name2]

                print(d1, d2, unit1, unit2)
                units = unit1 + ' ' + unit2
                name = name1 + ' ' + name2
                print(units, name)
                self.cumulants.add_variable(name,d1,d2,'symmetric',units)

        print('Sim', self.cumulants.list_of_names, np.shape(self.cumulants.values))
        print('!!!?',self.cumulants.get_dof('u p'))
        n = self.cumulants.get_dof('u p')
        print('?????',n, type(n), n[1])

        m = self.cumulants.get_dof('u p')[0]
        print('m',n,m,type(m))
        # Add Cumulants as Prognostic Variables
        # self.cumulants.add_variable('uu',0,0,'symmetric',units='(m s^1)^2')
        # self.cumulants.add_variable('uw',0,2,'symmetric',units='(m s^1)^2')
        # self.cumulants.add_variable('ww',2,2,'symmetric',units='(m s^1)^2')
        # self.cumulants.add_variable('ut',0,3,'symmetric',units='m s^1 K')
        # self.cumulants.add_variable('wt',2,3,'symmetric',units='m s^1 K')
        # self.cumulants.add_variable('tt',3,3,'symmetric',units='K^2')
        # self.cumulants.add_variable('up',0,4,'symmetric',units='m s^1 Pa')
        # self.cumulants.add_variable('wp',2,4,'symmetric',units='m s^1 Pa')
        # self.cumulants.add_variable('tp',3,4,'symmetric',units='K Pa')
        # self.cumulants.add_variable('pp',4,4,'symmetric',units='Pa^2')

        self.velocities_mean.initialize(self.grid)
        self.scalars_mean.initialize(self.grid)
        self.pressure_mean.initialize(self.grid)
        self.cumulants.initialize(self.grid)



        self.mean_momentumadvection.initialize(self.grid, self.velocities_mean)
        self.mean_momentumadvection.initialize(self.grid, self.velocities_mean)
    #
    #
    #     self.timestepping_manager.initialize(self.nml)      # ????
    #
    #     self.init.initialize(self.case_dict, self.grid, self.basicstate, self.velocities, self.scalars,self.io,self.comm)
    #
    #     self.scalars.update_boundary_conditions(self.grid,self.comm)
    #     self.velocities.update_boundary_conditions(self.grid,self.comm)
    #
    #
    #
    #     self.thermodynamics.initialize(self.grid,self.io)
    #
    #     self.radiation.initialize(self.case_dict,self.grid,self.scalars, self.thermodynamics,self.io)
    #
    #
    #
    #     self.forcing.initialize(self.case_dict,self.basicstate,self.grid,self.scalars,self.velocities,self.thermodynamics,self.io)
    #
    #     self.scalaradvection.initialize(self.nml,self.grid, self.scalars)
#         self.momentumadvection.initialize(self.grid, self.velocities)
    #
    #     self.surface.initialize(self.case_dict,self.comm,self.grid,self.basicstate,self.scalars,self.velocities,self.io)
    #
    #     self.poisson.initialize(self.grid)
    #
    #     self.damping.initialize(self.case_dict,self.grid)
    #
    #
    #
    #
    # def run(self):
    #     self.thermodynamics.update(self.grid,self.basicstate,self.scalars, self.velocities,self.comm,self.io)
    #     self.velocities.zero_all_tendencies()
    #     self.scalars.zero_all_tendencies()
    #
    #     while(self.timestepping.time < self.timestepping.timemax):
    #         tic = time.time()
    #         for self.timestepping.rk_step in range(self.timestepping.num_rk_step):
    #
    #             self.thermodynamics.update(self.grid,self.basicstate,self.scalars, self.velocities,self.comm,self.io)
    #
    #             self.forcing.update(self.grid,self.basicstate,self.scalars,self.velocities,self.thermodynamics,
    #                                 self.timestepping,self.comm,self.io)
    #
    #             self.surface.update(self.grid,self.basicstate,self.scalars,self.velocities,self.thermodynamics,self.radiation,self.timestepping,
    #                                 self.comm,self.io)
    #             self.scalaradvection.update( self.grid, self.basicstate, self.scalars, self.velocities)
    #             self.momentumadvection.update( self.grid, self.basicstate, self.scalars, self.velocities)
    #
    #             self.damping.update(self.grid,self.scalars,self.velocities)
    #
    #             self.timestepping.update(self.scalars,self.velocities,self.grid,self.io,self.comm)
    #
    #             self.poisson.update(self.comm,self.grid, self.basicstate, self.scalars, self.velocities, self.pressure,
    #                                    self.timestepping)
    #
    #             self.scalars.update_boundary_conditions(self.grid,self.comm)
    #             self.velocities.update_boundary_conditions(self.grid,self.comm)
    #
    #             self.timestepping_manager.update(self.grid,self.scalars,self.velocities,self.sgs,self.timestepping,self.comm,self.io)
    #
    #         toc = time.time()
    #
    #
    #
    # def finalize(self):
    #     pass
