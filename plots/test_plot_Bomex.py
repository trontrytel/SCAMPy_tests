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
def sim_data(request):

    # generate namelists and paramlists
    setup = pls.simulation_setup('Bomex')
    # change the defaults
    setup['namelist']['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True
    #setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['entrainment'] = "b_w2"
    #setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 5
    #setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['mixing_length'] = "sbl"
    #setup["namelist"]['turbulence']['EDMF_PrognosticTKE']["use_sommeria_deardorff"] = False

    #setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['use_steady_updrafts'] = True
    setup['namelist']['thermodynamics']['saturation'] = 'sa_quadrature'
    #setup['namelist']['thermodynamics']['saturation'] = 'sa_mean'

    #setup["paramlist"]['turbulence']['prandtl_number'] = 0.8
    #setup["paramlist"]['turbulence']['Ri_bulk_crit'] = 0.2

    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.2      # 0.1
    #setup['paramlist']['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 1.0   # 9.9
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.22   # 0.022
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.17     # 0.5

    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["detrainment_factor"] = 1
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["entrainment_factor"] = 1

    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["pressure_buoy_coeff"] = 1./3
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["pressure_drag_coeff"] = 0.375
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["pressure_plume_spacing"] = 500.0


    # run scampy
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(pls.removing_files)

    return sim_data

def test_plot_Bomex(sim_data):
    """
    plot Bomex profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, n_steps=100)

    pls.plot_mean(data_to_plot,   "Bomex_quicklook.pdf")
    pls.plot_drafts(data_to_plot, "Bomex_quicklook_drafts.pdf")

def test_plot_timeseries_Bomex(sim_data):
    """
    plot Bomex timeseries
    """
    data_to_plot = pls.read_data_srs(sim_data)

    pls.plot_timeseries(data_to_plot, "Bomex")

def test_plot_timeseries_1D_Bomex(sim_data):
    """
    plot Bomex 1D timeseries
    """
    data_to_plot = pls.read_data_timeseries(sim_data)

    pls.plot_timeseries_1D(data_to_plot, "Bomex_timeseries_1D.pdf")

def test_plot_var_covar_Bomex(sim_data):
    """
    plot Bomex var covar
    """
    data_to_plot = pls.read_data_avg(sim_data, n_steps=100, var_covar=True)

    pls.plot_var_covar_mean(data_to_plot,       "Bomex_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot, "Bomex_var_covar_components.pdf")
