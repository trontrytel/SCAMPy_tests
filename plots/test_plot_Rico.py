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
    # change the defaults
    setup['namelist']['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True
    setup['namelist']['thermodynamics']['saturation'] = 'sa_quadrature'
    #setup['namelist']['microphysics']['rain_model'] = False
    #setup['namelist']['microphysics']['max_supersaturation'] = 1e-4
    setup["paramlist"]['turbulence']['updraft_microphysics']['max_supersaturation'] = 1e-4


    setup['namelist']['time_stepping']['dt'] = 1.
    #setup['namelist']['time_stepping']['t_max'] = 8450. #86400.

    #setup['namelist']['stats_io']['frequency'] = 30.0

    setup['paramlist']['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1
    setup['paramlist']['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 9.9

    # run scampy
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

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
    pls.plot_var_covar_components(data_to_plot, "Rico_var_covar_components.pdf")

def test_plot_timeseries_Rico(sim_data):
    """
    plot timeseries
    """
    data_to_plot = pls.read_data_srs(sim_data)

    pls.plot_timeseries(data_to_plot, "Rico")

def test_plot_timeseries_1D_Rico(sim_data):
    """
    plot Rico 1D timeseries
    """
    data_to_plot = pls.read_data_timeseries(sim_data)

    pls.plot_timeseries_1D(data_to_plot, "Rico_timeseries_1D.pdf")

