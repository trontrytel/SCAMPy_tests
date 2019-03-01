import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../SCAMPy")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset

import pytest
import pprint as pp
import numpy as np

import main as scampy
import plot_scripts as pls

@pytest.fixture(scope="module")
def sim_data(request):

    # generate namelists and paramlists
    setup = pls.simulation_setup('Rico')

    # change the namelist parameters
    setup['namelist']['thermodynamics']['saturation'] = 'sa_quadrature'
    #setup['namelist']['thermodynamics']['saturation'] = 'sa_mean'

    setup['namelist']['microphysics']['rain_model']          = True
    setup['namelist']['microphysics']['rain_const_area']     = True
    setup['namelist']['microphysics']['max_supersaturation'] = 0.0001 #0.1 # 1e-4
    setup['namelist']['microphysics']['rain_area_value']     = 1.

    setup['namelist']['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var']        = True
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['updraft_number']         = 5
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['entrainment']            = "suselj"
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['mixing_length']          = "sbl"
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']["use_sommeria_deardorff"] = False
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['use_local_micro']        = True

    # change the paramlist parameters
    setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['surface_area']       = 0.1  # 0.2  # 0.1
    setup['paramlist']['turbulence']['EDMF_PrognosticTKE']['max_area_factor']    = 10.  # 1.0  # 9.9

    setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff']     = 1.0  # 0.22 # 2.
    setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff']       = 0.1  # 0.17 # 0.1

    setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["detrainment_factor"] = 0.75  # 1.
    setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["entrainment_factor"] = 0.75  # 1.

    # run scampy
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')
    #sim_data = Dataset("/Users/ajaruga/clones/SCAMPy/Output.Rico.6/stats/Stats.Rico.nc", 'r')

    # remove netcdf file after tests
    request.addfinalizer(pls.removing_files)

    return sim_data

def test_plot_Rico(sim_data):
    """
    plot Rico profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, n_steps=100)

    pls.plot_mean(data_to_plot,   "Rico_quicklook.pdf")
    pls.plot_drafts(data_to_plot, "Rico_quicklook_drafts.pdf")

def test_plot_var_covar_Rico(sim_data):
    """
    plot Rico variance and covariance of H and QT profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, n_steps=100, var_covar=True)

    pls.plot_var_covar_mean(data_to_plot,       "Rico_var_covar_mean.pdf")
    #pls.plot_var_covar_components(data_to_plot, "Rico_var_covar_components.pdf")

def test_plot_timeseries_Rico(sim_data):
    """
    plot timeseries
    """
    data_to_plot = pls.read_data_srs(sim_data)

    pls.plot_timeseries(data_to_plot, "Rico")

#def test_plot_timeseries_1D_Rico(sim_data):
#    """
#    plot Rico 1D timeseries
#    """
#    data_to_plot = pls.read_data_timeseries(sim_data)
#
#    pls.plot_timeseries_1D(data_to_plot, "Rico_timeseries_1D.pdf")
#
