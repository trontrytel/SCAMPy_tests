import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../SCAMPy")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset

import pytest
import numpy as np
import pprint as pp

import main as scampy
import plot_scripts as pls

@pytest.fixture(scope="module")
def res(request):

    # generate namelists and paramlists
    setup = pls.simulation_setup('Bomex')

    # change the defaults
    setup['namelist']['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['updraft_number']  = 1
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['mixing_length']   = "sbl"
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['entrainment']     = "b_w2"
    #setup['namelist']['turbulence']['EDMF_PrognosticTKE']['entrainment']    = "suselj"

    setup['namelist']['thermodynamics']['saturation'] = 'sa_quadrature'
    #setup['namelist']['thermodynamics']['saturation'] = 'sa_mean'

    setup["namelist"]['microphysics']['rain_model']           = True
    setup["namelist"]['microphysics']['rain_const_area']      = True
    setup["namelist"]['microphysics']['max_supersaturation']  = 0.1 # 1e-4
    setup['namelist']['microphysics']['rain_area_value']      = 1.

    setup['namelist']['grid']['dims'] = 1
    setup['namelist']['grid']['dz'] = 20.
    setup['namelist']['grid']['gw'] = 2
    setup['namelist']['grid']['nz'] = 150

    setup['namelist']['time_stepping']['t_max'] = 21600
    setup['namelist']['time_stepping']['dt'] = 10

    setup['namelist']["stats_io"]["frequency"] = 100.

    # additional parameters for offline runs
    scampifylist = {}
    scampifylist["offline"] = True
    scampifylist["nc_file"] = '../SCAMPY_tests/plots/BOMEX_comparison/Bomex_SA_LES_data/Stats.Bomex.nc'

    # TODO - compare with prescribed scalar var simulations
    #setup['namelist']['offline']['use_prescribed_scalar_var'] = True

    if scampifylist["offline"]:
        # run scampy offline
        print "offline run"
        #scampy.main_scampify(setup["namelist"], setup["paramlist"], scampifylist)
    else:
        # run scampy online
        print "online run"
        scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    SCM_data = Dataset(setup["outfile"], 'r')
    LES_data = Dataset(scampifylist["nc_file"], 'r')

    # remove netcdf file after tests
    #request.addfinalizer(pls.removing_files)

    res = {"SCM_data": SCM_data,
           "LES_data": LES_data}

    return res

def test_plot_Bomex(res):
    """
    plot Bomex profiles
    """
    SCM_data_avg = pls.read_data_avg(res["SCM_data"], n_steps=50)
    SCM_data_srs = pls.read_data_srs(res["SCM_data"])

    LES_data_avg = pls.read_LES_data_avg(res["LES_data"], n_steps=50)
    LES_data_srs = pls.read_LES_data_srs(res["LES_data"])

    pls.plot_mean(SCM_data_avg,   "Bomex_quicklook.pdf")
    pls.plot_drafts(SCM_data_avg, "Bomex_quicklook_drafts.pdf")

    pls.plot_drafts_area_wght(SCM_data_avg, SCM_data_srs, LES_data_avg, LES_data_srs, "Bomex_quicklook_drafts_wght.pdf")

    pls.plot_percentiles(LES_data_srs, SCM_data_avg, "Bomex_percentiles.pdf")

#def test_plot_timeseries_Bomex(res):
#    """
#    plot Bomex timeseries
#    """
#    data_to_plot = pls.read_data_srs(res["SCM_data"])
#
#    pls.plot_timeseries(data_to_plot, "Bomex")
#
#def test_plot_timeseries_1D_Bomex(res):
#    """
#    plot Bomex 1D timeseries
#    """
#    data_to_plot = pls.read_data_timeseries(res["SCM_data"])
#
#    pls.plot_timeseries_1D(data_to_plot, "Bomex_timeseries_1D.pdf")
#
#def test_plot_var_covar_Bomex(res):
#    """
#    plot Bomex var covar
#    """
#    data_to_plot = pls.read_data_avg(res["SCM_data"], n_steps=100, var_covar=True)
#
#    pls.plot_var_covar_mean(data_to_plot,       "Bomex_var_covar_mean.pdf")
#    #pls.plot_var_covar_components(data_to_plot, "Bomex_var_covar_components.pdf")
