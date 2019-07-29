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
    setup = pls.simulation_setup('TRMM_LBA')

    # chenge the defaults
    setup['namelist']['thermodynamics']['saturation'] = 'sa_quadrature'

    setup['namelist']['microphysics']['rain_model'] = True
    setup['namelist']['microphysics']['rain_const_area'] = True
    #setup['namelist']['microphysics']['rain_area_value'] = 1.
    setup['namelist']['microphysics']['max_supersaturation'] = 0.01  # 1e-4

    # run scampy
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(pls.removing_files)

    return sim_data

#@pytest.mark.skip(reason="deep convection not working with current defaults")
def test_plot_TRMM_LBA(sim_data):
    """
    plot TRMM_LBA profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, n_steps=100)

    pls.plot_mean(data_to_plot,   "TRMM_LBA_quicklook.pdf")
    pls.plot_drafts(data_to_plot, "TRMM_LBA_quicklook_drafts.pdf")

#@pytest.mark.skip(reason="deep convection not working with current defaults")
def test_plot_timeseries_TRMM_LBA(sim_data):
    """
    plot timeseries
    """
    data_to_plot = pls.read_data_srs(sim_data)

    pls.plot_timeseries(data_to_plot, "TRMM_LBA")

#@pytest.mark.skip(reason="deep convection not working with current defaults")
def test_plot_timeseries_1D_TRMM_LBA(sim_data):
    """
    plot TRMM_LBA 1D timeseries
    """
    data_to_plot = pls.read_data_timeseries(sim_data)

    pls.plot_timeseries_1D(data_to_plot, "TRMM_LBA_timeseries_1D.pdf")

#@pytest.mark.skip(reason="deep convection not working with current defaults")
def test_plot_var_covar_TRMM_LBA(sim_data):
    """
    plot TRMM LBA var covar
    """
    data_to_plot = pls.read_data_avg(sim_data, n_steps=100, var_covar=True)

    pls.plot_var_covar_mean(data_to_plot,       "TRMM_LBA_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot, "TRMM_LBA_var_covar_components.pdf")
