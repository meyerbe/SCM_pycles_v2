#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import netCDF4 as nc
import os
import shutil
# cimport ParallelMPI
# cimport TimeStepping
cimport PrognosticVariables
# cimport DiagnosticVariables
cimport Grid
import numpy as np
cimport numpy as np
import cython

cdef class NetCDFIO_Stats:
    def __init__(self):
        self.root_grp = None
        self.profiles_grp = None
        self.ts_grp = None
        return

    @cython.wraparound(True)
    cpdef initialize(self, dict namelist, Grid.Grid Gr):
        print('StatsIO.initialize', Gr.nz)
    # cpdef initialize(self, dict namelist, Gr):
        self.last_output_time = 0.0
        self.uuid = str(namelist['meta']['uuid'])
        self.frequency = namelist['stats_io']['frequency']

        # Setup the statistics output path
        outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + self.uuid[-5:]))
        print(outpath)

        try:
            os.mkdir(outpath)
        except:
            pass

        self.stats_path = str( os.path.join(outpath, namelist['stats_io']['stats_dir']))
        try:
            os.mkdir(self.stats_path)
        except:
            pass

        # Setup the restart repository
        self.path_plus_file = str( self.stats_path + '/' + 'Stats.' + namelist['meta']['simname'] + '.nc')
        # if os.path.exists(self.path_plus_file):
        #     for i in range(100):
        #         res_name = 'Restart_'+str(i)
        #         print "Here " + res_name
        #         if os.path.exists(self.path_plus_file):
        #             self.path_plus_file = str( self.stats_path + '/' + 'Stats.' + namelist['meta']['simname']
        #                    + '.' + res_name + '.nc')
        #         else:
        #             break

        shutil.copyfile(
                os.path.join( './', namelist['meta']['simname'] + '.in'),
                os.path.join( outpath, namelist['meta']['simname'] + '.in'))
        self.setup_stats_file(Gr)
        return
#






    cpdef open_files(self):
        self.root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        self.profiles_grp = self.root_grp.groups['profiles']
        self.ts_grp = self.root_grp.groups['timeseries']
        return

    cpdef close_files(self):
        self.root_grp.close()
        return

    cpdef setup_stats_file(self, Grid.Grid Gr):
        print('NetCDFIO_Stats: setup_stats_file')

        root_grp = nc.Dataset(self.path_plus_file, 'w', format='NETCDF4')

        # Set profile dimensions
        profile_grp = root_grp.createGroup('profiles')
        profile_grp.createDimension('z', Gr.nz)
        profile_grp.createDimension('t', None)
        z = profile_grp.createVariable('z', 'f8', ('z'))
        # z[:] = np.array(Gr.z[Gr.gw:-Gr.gw])
        z_half = profile_grp.createVariable('z_half', 'f8', ('z'))
#         z_half[:] = np.array(Gr.z_half[Gr.dims.gw:-Gr.dims.gw])
        profile_grp.createVariable('t', 'f8', ('t'))
#         del z
#         del z_half

        reference_grp = root_grp.createGroup('reference')
        reference_grp.createDimension('z', Gr.nz)
        z = reference_grp.createVariable('z', 'f8', ('z'))
        z[:] = np.array(Gr.z[Gr.gw:-Gr.gw])
        z_half = reference_grp.createVariable('z_half', 'f8', ('z'))
        z_half[:] = np.array(Gr.z_half[Gr.gw:-Gr.gw])
        del z
        del z_half

        ts_grp = root_grp.createGroup('timeseries')
        ts_grp.createDimension('t', None)
        ts_grp.createVariable('t', 'f8', ('t'))

        root_grp.close()
        return




    '''adding and writing data'''
    cpdef add_profile(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        profile_grp = root_grp.groups['profiles']
        new_var = profile_grp.createVariable(var_name, 'f8', ('t', 'z'))

        root_grp.close()
        return

    cpdef add_reference_profile(self, var_name, Grid.Grid Gr):
        '''
        Adds a profile to the reference group NetCDF Stats file.
        :param var_name: name of variable
        :param Gr: Grid class
        :return:
        '''
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        reference_grp = root_grp.groups['reference']
        new_var = reference_grp.createVariable(var_name, 'f8', ('z',))

        root_grp.close()
        return

    cpdef add_ts(self, var_name, Grid.Grid Gr):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        ts_grp = root_grp.groups['timeseries']
        new_var = ts_grp.createVariable(var_name, 'f8', ('t',))

        root_grp.close()
        return

    cpdef write_profile(self, var_name, double[:] data):
        #root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        #profile_grp = root_grp.groups['profiles']
        var = self.profiles_grp.variables[var_name]
        var[-1, :] = np.array(data)
        #root_grp.close()
        return

    cpdef write_reference_profile(self, var_name, double[:] data):
        '''
        Writes a profile to the reference group NetCDF Stats file. The variable must have already been
        added to the NetCDF file using add_reference_profile
        :param var_name: name of variables
        :param data: data to be written to file
        :return:
        '''
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        reference_grp = root_grp.groups['reference']
        var = reference_grp.variables[var_name]
        var[:] = np.array(data)
        root_grp.close()
        return

    @cython.wraparound(True)
    cpdef write_ts(self, var_name, double data):
            #root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
            #ts_grp = root_grp.groups['timeseries']
        var = self.ts_grp.variables[var_name]
        var[-1] = data
            #root_grp.close()
        return

    cpdef write_simulation_time(self, double t):
        #root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        #profile_grp = root_grp.groups['profiles']
        #ts_grp = root_grp.groups['timeseries']

        # Write to profiles group
        profile_t = self.profiles_grp.variables['t']
        profile_t[profile_t.shape[0]] = t

        # Write to timeseries group
        ts_t = self.ts_grp.variables['t']
        ts_t[ts_t.shape[0]] = t

        #root_grp.close()
        return





cdef class NetCDFIO_CondStats:
    def __init__(self):

        return
#
#     @cython.wraparound(True)
#     cpdef initialize(self, dict namelist, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):
#
#         self.last_output_time = 0.0
#         self.uuid = str(namelist['meta']['uuid'])
#         # if a frequency is not defined for the conditional statistics, set frequency to the maximum simulation time
#         try:
#             self.frequency = namelist['conditional_stats']['frequency']
#         except:
#             self.frequency = namelist['time_stepping']['t_max']
#
#
#         # Setup the statistics output path
#         outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.' + self.uuid[-5:]))
#
#         if Pa.rank == 0:
#             try:
#                 os.mkdir(outpath)
#             except:
#                 pass
#
#         # Set a default name for the output directory if it is not defined in the namelist
#         try:
#             self.stats_path = str( os.path.join(outpath, namelist['conditional_stats']['stats_dir']))
#         except:
#             self.stats_path = str( os.path.join(outpath, 'cond_stats'))
#
#         if Pa.rank == 0:
#             try:
#                 os.mkdir(self.stats_path)
#             except:
#                 pass
#
#
#         self.path_plus_file = str( self.stats_path + '/' + 'CondStats.' + namelist['meta']['simname'] + '.nc')
#         if os.path.exists(self.path_plus_file):
#             for i in range(100):
#                 res_name = 'Restart_'+str(i)
#                 if os.path.exists(self.path_plus_file):
#                     self.path_plus_file = str( self.stats_path + '/' + 'CondStats.' + namelist['meta']['simname']
#                            + '.' + res_name + '.nc')
#                 else:
#                     break
#
#         Pa.barrier()
#
#
#
#         if Pa.rank == 0:
#             shutil.copyfile(
#                 os.path.join( './', namelist['meta']['simname'] + '.in'),
#                 os.path.join( outpath, namelist['meta']['simname'] + '.in'))
#         return
#
#     cpdef create_condstats_group(self, str groupname, str dimname, double [:] dimval, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):
#
#         if Pa.rank == 0:
#             root_grp = nc.Dataset(self.path_plus_file, 'w', format='NETCDF4')
#             sub_grp = root_grp.createGroup(groupname)
#             sub_grp.createDimension('z', Gr.dims.n[2])
#             sub_grp.createDimension(dimname, len(dimval))
#             sub_grp.createDimension('t', None)
#             z = sub_grp.createVariable('z', 'f8', ('z'))
#             z[:] = np.array(Gr.z[Gr.dims.gw:-Gr.dims.gw])
#             dim = sub_grp.createVariable(dimname, 'f8', (dimname))
#             dim[:] = np.array(dimval[:])
#             sub_grp.createVariable('t', 'f8', ('t'))
#             del z
#             del dim
#             root_grp.close()
#         return
#
#     cpdef add_condstat(self, str varname, str groupname, str dimname, Grid.Grid Gr, ParallelMPI.ParallelMPI Pa):
#
#         if Pa.rank == 0:
#             root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
#             sub_grp = root_grp.groups[groupname]
#             new_var = sub_grp.createVariable(varname, 'f8', ('t', 'z', dimname))
#
#             root_grp.close()
#
#         return
#
#
#     cpdef write_condstat(self, varname, groupname, double [:,:] data, ParallelMPI.ParallelMPI Pa):
#         if Pa.rank == 0:
#             root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
#             sub_grp = root_grp.groups[groupname]
#             var = sub_grp.variables[varname]
#
#             var[-1, :,:] = np.array(data)[:,:]
#
#             root_grp.close()
#         return
#
#
#     cpdef write_condstat_time(self, double t, ParallelMPI.ParallelMPI Pa):
#         if Pa.rank == 0:
#             try:
#                 root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
#                 for groupname in root_grp.groups:
#                     sub_grp = root_grp.groups[groupname]
#
#                     # Write to sub_grp
#                     group_t = sub_grp.variables['t']
#                     group_t[group_t.shape[0]] = t
#
#                 root_grp.close()
#             except:
#                 pass
#         return