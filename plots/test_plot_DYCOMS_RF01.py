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
def res(request):

    # generate namelists and paramlists
    setup = pls.simulation_setup('DYCOMS_RF01')
    # chenge the defaults
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['entrainment'] = "suselj"
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['mixing_length'] = "sbl"

    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.2
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.022
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.5
    #setup["paramlist"]['turbulence']['prandtl_number'] = 0.8
    #setup["paramlist"]['turbulence']['Ri_bulk_crit'] = 0.2

    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    setup['namelist']['thermodynamics']['saturation'] = 'sa_quadrature'
    #setup['namelist']['thermodynamics']['saturation'] = 'sa_mean'
    setup['namelist']['microphysics']['rain_model'] = True
    setup['namelist']['microphysics']['rain_const_area'] = True
    #setup['namelist']['microphysics']['rain_area_value'] = 1.
    setup['namelist']['microphysics']['max_supersaturation'] = 1e3 #0.1  # 1e-4

    #setup["paramlist"]['turbulence']['prandtl_number'] = 0.8
    #setup["paramlist"]['turbulence']['Ri_bulk_crit'] = 0.2

    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.2      # 0.1
    #setup['paramlist']['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 1.0   # 9.9
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.22   # 0.022
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.17     # 0.5
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["detrainment_factor"] = .5 #1. #0.5
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["entrainment_factor"] = .5 #1. #0.5
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["pressure_buoy_coeff"] = 1./3
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["pressure_drag_coeff"] = 0.375
    #setup["paramlist"]['turbulence']['EDMF_PrognosticTKE']["pressure_plume_spacing"] = 500.0

    setup['namelist']['grid']['dims'] = 1
    setup['namelist']['grid']['dz'] = 5.
    setup['namelist']['grid']['gw'] = 2
    setup['namelist']['grid']['nz'] = 300

    setup['namelist']['time_stepping']['t_max'] = 14400
    setup['namelist']['time_stepping']['dt'] = 4

    setup['namelist']["stats_io"]["frequency"] = 60.

    # additional parameters for offline runs
    scampifylist = {}
    scampifylist["offline"] = True
    scampifylist["nc_file"] = '../SCAMPY_tests/plots/DYCOMS_comparison/DYCOMS_SA_LES_data_new/Stats.DYCOMS_RF01.nc'
    scampifylist["les_stats_freq"] = 60.

    # TODO - compare with prescribed scalar var simulations
    #setup['namelist']['offline']['use_prescribed_scalar_var'] = True

    if scampifylist["offline"]:
        # run scampy offline
        print "offline run"
        scampy.main_scampify(setup["namelist"], setup["paramlist"], scampifylist)
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

def test_plot_DYCOMS_RF01(res):
    """
    plot DYCOMS_RF01 quicklook profiles
    """
    SCM_data_avg = pls.read_data_avg(res["SCM_data"], n_steps=30)
    SCM_data_srs = pls.read_data_srs(res["SCM_data"])

    LES_data_avg = pls.read_LES_data_avg(res["LES_data"], 'Dycoms_RF01', n_steps=30)
    LES_data_srs = pls.read_LES_data_srs(res["LES_data"])

    pls.plot_mean(SCM_data_avg,   "Dycoms_RF01_quicklook.pdf")
    pls.plot_drafts(SCM_data_avg, "Dycoms_RF01_quicklook_drafts.pdf")

    pls.plot_drafts_area_wght(SCM_data_srs, LES_data_srs, "Dycoms_RF01_quicklook_drafts_wght.pdf", "Dycoms_RF01", avg_step=30)

    pls.plot_percentiles(LES_data_srs, SCM_data_avg, "Dycoms_RF01_percentiles.pdf", 'Dycoms_RF01')

#def test_plot_var_covar_DYCOMS_RF01(sim_data):
#    """
#    plot DYCOMS_RF01 quicklook profiles
#    """
#    data_to_plot = pls.read_data_avg(sim_data, n_steps=100, var_covar=True)
#
#    pls.plot_var_covar_mean(data_to_plot,       "DYCOMS_RF01_var_covar_mean.pdf")
#    pls.plot_var_covar_components(data_to_plot, "DYCOMS_RF01_var_covar_components.pdf")
#
#def test_plot_timeseries_DYCOMS(sim_data):
#    """
#    plot timeseries
#    """
#    data_to_plot = pls.read_data_srs(sim_data)
#
#    pls.plot_timeseries(data_to_plot, "DYCOMS")
#
#def test_plot_timeseries_1D_DYCOMS_RF01(sim_data):
#    """
#    plot DYCOMS_RF01 1D timeseries
#    """
#    data_to_plot = pls.read_data_timeseries(sim_data)
#
#    pls.plot_timeseries_1D(data_to_plot, "DYCOMS_RF01_timeseries_1D.pdf")
#
#def test_DYCOMS_RF01_radiation(sim_data):
#    """
#    plots DYCOMS_RF01
#    """
#    import matplotlib as mpl
#    mpl.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
#    import matplotlib.pyplot as plt
#
#    fig = plt.figure(1)
#    fig.set_figheight(12)
#    fig.set_figwidth(14)
#    mpl.rcParams.update({'font.size': 18})
#    mpl.rc('lines', linewidth=4, markersize=10)
#
#    plt_data = pls.read_data_avg(sim_data,     n_steps=100, var_covar=False)
#    rad_data = pls.read_rad_data_avg(sim_data, n_steps=100)
#
#    plots = []
#    # loop over simulation and reference data for t=0 and t=-1
#    x_lab  = ['longwave radiative flux [W/m2]', 'dTdt [K/day]',       'QT [g/kg]',         'QL [g/kg]']
#    legend = ["lower right",                    "lower left",         "lower left",        "lower right"]
#    line   = ['--',                             '--',                 '-',                 '-']
#    plot_y = [rad_data["rad_flux"],             rad_data["rad_dTdt"], plt_data["qt_mean"], plt_data["ql_mean"]]
#    plot_x = [rad_data["z"],                    plt_data["z_half"],   plt_data["z_half"],  plt_data["z_half"]]
#    color  = ["palegreen",                      "forestgreen"]
#    label  = ["ini",                            "end"        ]
#
#    for plot_it in range(4):
#        plots.append(plt.subplot(2,2,plot_it+1))
#                              #(rows, columns, number)
#        for it in range(2):
#            plots[plot_it].plot(plot_y[plot_it][it], plot_x[plot_it], '.-', color=color[it], label=label[it])
#        plots[plot_it].legend(loc=legend[plot_it])
#        plots[plot_it].set_xlabel(x_lab[plot_it])
#        plots[plot_it].set_ylabel('z [m]')
#    plots[2].set_xlim([1, 10])
#    plots[3].set_xlim([-0.1, 0.5])
#
#    plt.savefig("plots/output/DYCOMS_RF01_radiation.pdf")
#    plt.clf()
#
