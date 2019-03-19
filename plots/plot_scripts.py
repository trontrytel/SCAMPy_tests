import sys
sys.path.insert(0, "./")
sys.path.insert(0, "./../SCAMPY/")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset
import numpy as np
import pprint as pp

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm

import scampy_wrapper as wrp

def simulation_setup(case):
    """
    generate namelist and paramlist files for scampy
    choose the name of the output folder
    """
    # Filter annoying Cython warnings that serve no good purpose.
    # see https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

    # simulation related parameters
    os.system("python ../SCAMPy/generate_namelist.py " + case)
    file_case = open(case + '.in').read()
    # turbulence related parameters
    os.system("python ../SCAMPy/generate_paramlist.py " +  case)
    file_params = open('paramlist_' + case + '.in').read()

    namelist  = json.loads(file_case)
    paramlist = json.loads(file_params)

    namelist['output']['output_root'] = "./Tests."
    namelist['meta']['uuid'] = case
    # TODO - copied from NetCDFIO
    # ugly way to know the name of the folder where the data is saved
    uuid = str(namelist['meta']['uuid'])
    outpath = str(
        os.path.join(
            namelist['output']['output_root'] +
            'Output.' +
            namelist['meta']['simname'] +
            '.' +
            uuid[len(uuid )-5:len(uuid)]
        )
    )

    outfile = outpath + "/stats/Stats." + case + ".nc"

    res = {"namelist"  : namelist,
           "paramlist" : paramlist,
           "outfile"   : outfile}
    return res


def removing_files():
    """
    remove the folder with netcdf files from tests
    """
    #TODO - think of something better
    cmd = "rm -r Tests.Output.*"
    subprocess.call(cmd , shell=True)


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Taken from https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def read_data_avg(sim_data, n_steps=0, var_covar=False):
    """
    Read in the data from netcdf file into a dictionary that can be used for plots

    Input:
    sim_data  - netcdf Dataset with simulation results
    n_steps   - number of timesteps to average over
    var_covar - flag for when we also want to read in the variance and covariance fields
    """
    variables = ["temperature_mean", "thetal_mean", "qt_mean", "ql_mean", "qr_mean",\
                 "updraft_area", "env_qt", "updraft_qt", "env_ql", "updraft_ql",\
                 "env_qr", "updraft_qr", "updraft_w", "env_w"]

    variables_var = ["Hvar_mean", "QTvar_mean", "HQTcov_mean", "env_Hvar", "env_QTvar", "env_HQTcov"]

    if var_covar:
        variables.extend(variables_var)

    # read the data from t=init and t=end
    data_to_plot = {}
    time = [0, -1]
    for var in variables:
        data_to_plot[var] = []
        for it in range(2):
            if ("buoyancy" in var):
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]) * 10000) #cm2/s3
            elif ("qt" in var or "ql" in var or "qr" in var):
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]) * 1000)  #g/kg
            else:
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]))

    # TODO - should be fixed so that I divide by the number of times where an updraft was present
    # and not the total number of timesteps
    # this matters for the places with small to non-existend updraft fractions
    # add averaging over last n_steps timesteps
    if(n_steps > 0):
        for var in variables:
            for time_it in range(-2, -1*n_steps-1, -1):
                if ("buoyancy" in var):
                    data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :]) * 10000  #cm2/s3
                elif ("qt" in var or "ql" in var or "qr" in var):
                    data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :]) * 1000   #g/kg
                else:
                    data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :])

            data_to_plot[var][1] /= n_steps

    # read reference data
    data_to_plot["z_half"] = np.array(sim_data["reference/z_half"][:])
    data_to_plot["p0_half"] = np.array(sim_data["reference/p0_half"][:])

    return data_to_plot

def read_LES_data_avg(sim_data, simname, n_steps=0):
    """
    Read in the data from netcdf file into a dictionary that can be used for plots

    Input:
    sim_data  - netcdf Dataset with simulation results
    n_steps   - number of timesteps to average over
    """

    # read LES tracers output
    variables = ["temperature_mean", "thetali_mean", "qt_mean", "ql_mean",\
                 "updraft_fraction", "env_qt", "updraft_qt", "env_ql", "updraft_ql",\
                 "updraft_w", "env_w"]

    # read the data from t=init and t=end
    data_to_plot = {}
    time = [0, -1]
    for var in variables:
        data_to_plot[var] = []
        for it in range(2):
            if ("buoyancy" in var):
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]) * 10000) #cm2/s3
            elif ("qt" in var or "ql" in var or "qr" in var):
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]) * 1000)  #g/kg
            else:
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]))

    # add averaging over last n_steps timesteps
    # TODO - see todo about averaging for Scampy
    if(n_steps > 0):
        for var in variables:
            for time_it in range(-2, -1*n_steps-1, -1):
                if ("buoyancy" in var):
                    data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :]) * 10000  #cm2/s3
                elif ("qt" in var or "ql" in var or "qr" in var):
                    data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :]) * 1000   #g/kg
                else:
                    data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :])

            data_to_plot[var][1] /= n_steps

    # read reference data
    if simname == 'Bomex':
        data_to_plot["z_half"]  = np.array(sim_data["reference/z_half"][:])
        data_to_plot["p0_half"] = np.array(sim_data["reference/p0_half"][:])
    else:
        data_to_plot["z_half"]  = np.array(sim_data["reference/z"][:])
        data_to_plot["p0_half"] = np.array(sim_data["reference/p0"][:])

    return data_to_plot

def read_rad_data_avg(sim_data, n_steps=0):
    """
    Read in the radiation forcing data from netcdf files into a dictionary that can be used for plots
    (so far its only applicable to the DYCOMS radiative forcing)

    Input:
    sim_data - netcdf Dataset with simulation results
    n_steps  - number of timesteps to average over
    """
    variables = ["rad_flux", "rad_dTdt"]

    time = [0, -1]
    rad_data = {"z" : np.array(sim_data["profiles/z"][:])}
    for var in variables:
        rad_data[var] = []
        for it in range(2):
            if ("rad_dTdt" in var):
                rad_data[var].append(np.array(sim_data["profiles/" + var][time[it], :]) * 60 * 60 * 24) # K/day
            else:
                rad_data[var].append(np.array(sim_data["profiles/" + var][time[it], :]))

    # add averaging over last n_steps timesteps
    # TODO - see TODO about averaging
    if(n_steps > 0):
        for var in variables:
            for time_it in range(-2, -1*n_steps-1, -1):
                if ("rad_dTdt" in var):
                    rad_data[var][1] += np.array(sim_data["profiles/" + var][time_it, :] * 60 * 60 * 24) # K/day
                else:
                    rad_data[var][1] += np.array(sim_data["profiles/" + var][time_it, :])

            rad_data[var][1] /= n_steps

    return rad_data


def read_data_srs(sim_data, var_covar=False):
    """
    Read in the data from netcdf file into a dictionary that can be used for timeseries of profiles plots

    Input:
    sim_data  - netcdf Dataset with simulation results
    var_covar - flag for when we also want to read in the variance and covariance fields
    """
    variables = ["temperature_mean", "thetal_mean", "qt_mean", "ql_mean", "qr_mean",\
                 "updraft_area", "env_qt", "updraft_qt", "env_ql", "updraft_ql", "updraft_thetal",\
                 "env_qr", "updraft_qr", "updraft_rain_area", "env_rain_area", "updraft_w", "env_w", "env_thetal"]

    variables_var = ["Hvar_mean", "QTvar_mean", "HQTcov_mean", "env_Hvar", "env_QTvar", "env_HQTcov"]
    if var_covar:
        variables.extend(variables_var)

    # read the data
    data_to_plot = {"z_half" : np.array(sim_data["profiles/z_half"][:]), "t" : np.array(sim_data["profiles/t"][:])}

    for var in variables:
        data_to_plot[var] = []
        if ("buoyancy" in var):
            data_to_plot[var] = np.transpose(np.array(sim_data["profiles/"  + var][:, :])) * 10000 #cm2/s3
        elif ("qt" in var or "ql" in var or "qr" in var):
            data_to_plot[var] = np.transpose(np.array(sim_data["profiles/"  + var][:, :])) * 1000  #g/kg
        elif ("p0" in var):
            data_to_plot[var] = np.transpose(np.array(sim_data["reference/" + var][:, :])) * 100   #hPa
        else:
            data_to_plot[var] = np.transpose(np.array(sim_data["profiles/"  + var][:, :]))

    return data_to_plot

def read_LES_data_srs(sim_data):
    """
    Read in the data from netcdf file into a dictionary that can be used for timeseries of profiles plots

    Input:
    sim_data  - netcdf Dataset with simulation results
    """
    variables = ["temperature_mean", "thetali_mean", "qt_mean", "ql_mean",\
                 "updraft_fraction", "env_qt", "updraft_qt", "env_ql", "updraft_ql",\
                 "updraft_thetali", "env_thetali",\
                 "updraft_w", "env_w"]

    # read the data
    data_to_plot = {"z_half" : np.array(sim_data["profiles/z_half"][:]), "t" : np.array(sim_data["profiles/t"][:])}

    for var in variables:
        data_to_plot[var] = []
        if ("buoyancy" in var):
            data_to_plot[var] = np.transpose(np.array(sim_data["profiles/"  + var][:, :])) * 10000 #cm2/s3
        elif ("qt" in var or "ql" in var or "qr" in var):
            data_to_plot[var] = np.transpose(np.array(sim_data["profiles/"  + var][:, :])) * 1000  #g/kg
        elif ("p0" in var):
            data_to_plot[var] = np.transpose(np.array(sim_data["reference/" + var][:, :])) * 100   #hPa
        else:
            data_to_plot[var] = np.transpose(np.array(sim_data["profiles/"  + var][:, :]))

    data_to_plot['env_Hvar']   = []
    data_to_plot['env_QTvar']  = []
    data_to_plot['env_HQTcov'] = []

    data_to_plot['upd_Hvar']   = []
    data_to_plot['upd_QTvar']  = []
    data_to_plot['upd_HQTcov'] = []

    data_to_plot['env_Hvar']   = np.transpose(np.array(sim_data["profiles/env_thetali2"][:,:])   -  np.array(sim_data["profiles/env_thetali"][:,:]) * np.array(sim_data["profiles/env_thetali"][:,:]))
    data_to_plot['env_QTvar']  = np.transpose(np.array(sim_data["profiles/env_qt2"][:,:])        -  np.array(sim_data["profiles/env_qt"][:,:])      * np.array(sim_data["profiles/env_qt"][:,:])) * 1e6
    data_to_plot['env_HQTcov'] = np.transpose(np.array(sim_data["profiles/env_qt_thetali"][:,:]) -  np.array(sim_data["profiles/env_thetali"][:,:]) * np.array(sim_data["profiles/env_qt"][:,:])) * 1e3

    data_to_plot['upd_Hvar']   = np.transpose(np.array(sim_data["profiles/updraft_thetali2"][:,:])   -  np.array(sim_data["profiles/updraft_thetali"][:,:]) * np.array(sim_data["profiles/updraft_thetali"][:,:]))
    data_to_plot['upd_QTvar']  = np.transpose(np.array(sim_data["profiles/updraft_qt2"][:,:])        -  np.array(sim_data["profiles/updraft_qt"][:,:])      * np.array(sim_data["profiles/updraft_qt"][:,:])) * 1e6
    data_to_plot['upd_HQTcov'] = np.transpose(np.array(sim_data["profiles/updraft_qt_thetali"][:,:]) -  np.array(sim_data["profiles/updraft_thetali"][:,:]) * np.array(sim_data["profiles/updraft_qt"][:,:])) * 1e3

    return data_to_plot

def read_data_timeseries(sim_data):
    """
    Read in the data from netcdf file into a dictionary that can be used for timeseries plots

    Input:
    sim_data - netcdf Dataset with simulation results
    """
    variables = ["updraft_cloud_cover", "updraft_cloud_base", "updraft_cloud_top", "updraft_lwp",\
                 "ustar", "shf", "lhf", "Tsurface"]

    # read the data
    data_to_plot = {"z_half" : np.array(sim_data["profiles/z_half"][:]), "t" : np.array(sim_data["profiles/t"][:])}

    for var in variables:
        data_to_plot[var] = []
        data_to_plot[var] = np.array(sim_data["timeseries/" + var][:])

    return data_to_plot


def plot_mean(data, title, folder="plots/output/"):
    """
    Plots mean profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    # read data
    qv_mean = np.array(data["qt_mean"]) - np.array(data["ql_mean"])

    # data to plot
    x_lab  = ['QV [g/kg]', 'QL [g/kg]',      'QR [g/kg]']
    plot_x = [qv_mean,      data["ql_mean"],  data["qr_mean"]]
    color  = ["palegreen", "forestgreen"]
    label  = ["ini", "end"]

    # iteration over plots
    plots = []
    for plot_it in range(3):
        plots.append(plt.subplot(1,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        for it in range(2): #init, end
            plots[plot_it].plot(plot_x[plot_it][it], data["z_half"], '.-', color=color[it], label=label[it])

    plots[0].legend()
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_drafts(data, title, folder="plots/output/"):
    """
    Plots updraft and environment profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    # read data
    qv_mean    = np.array(data["qt_mean"])    - np.array(data["ql_mean"])
    env_qv     = np.array(data["env_qt"])     - np.array(data["env_ql"])
    updraft_qv = np.array(data["updraft_qt"]) - np.array(data["updraft_ql"])

    # data to plot
    x_lab    = ["QV [g/kg]", "QL [g/kg]",        "QR [g/kg]",        "updraft area [%]"]
    plot_upd = [updraft_qv,  data["updraft_ql"], data["updraft_qr"], data["updraft_area"]]
    plot_env = [env_qv,      data["env_ql"],     data["env_qr"]]
    plot_mean= [qv_mean,     data["ql_mean"],    data["qr_mean"]]

    # iteration over plots
    plots = []
    for plot_it in range(4):
        plots.append(plt.subplot(2,2,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        #plot updrafts
        if (plot_it != 3):
            plots[plot_it].plot(plot_upd[plot_it][1], data["z_half"], ".-", color="blue", label="upd")
        if (plot_it == 3):
            plots[plot_it].plot(plot_upd[plot_it][1] * 100, data["z_half"], ".-", color="blue", label="upd")
        # plot environment
        if (plot_it < 3):
            plots[plot_it].plot(plot_env[plot_it][1], data["z_half"], ".-", color="red", label="env")
        # plot mean
        if (plot_it < 3):
            plots[plot_it].plot(plot_mean[plot_it][1], data["z_half"], ".-", color="purple", label="mean")

    plots[0].legend()
    plt.savefig(folder + title)
    plt.clf()

def plot_drafts_area_wght(scm_data_srs, les_data_srs, title, simname, avg_step=30, folder="plots/output/"):
    """
    Plots updraft and environment area weighted profiles from Scampy and LES data

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    if simname == 'Bomex':
        focus_z = [12, 27, 42, 62,  75, 90]
        x_range = [-.001, .015]
    elif simname == 'Dycoms_RF01':
        focus_z = [120, 130, 140, 150,  160, 170]
        x_range = [-.01, .38]
    else:
        print "Unrecognized simulation name"
        assert(False)

    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(12)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=3, markersize=6)

    # read data
    scm_srs_dict = {}
    scm_avg_dict = {}
    les_srs_dict = {}
    les_avg_dict = {}

    height = np.array(scm_data_srs["z_half"][:])

    scm_srs_dict["upd_area"] =  np.array(scm_data_srs["updraft_area"])[:,:]
    scm_srs_dict["env_area"] = 1. - scm_srs_dict["upd_area"]
    les_srs_dict["upd_area"] =  np.array(les_data_srs["updraft_fraction"])[:,:]
    les_srs_dict["env_area"] = 1. - les_srs_dict["upd_area"]

    scm_srs_dict["mean_ql"] = np.array(scm_data_srs["ql_mean"])[:,:]
    les_srs_dict["mean_ql"] = np.array(les_data_srs["ql_mean"])[:,:]

    scm_srs_dict["upd_ql"] = np.array(scm_data_srs["updraft_ql"])[:,:]
    scm_srs_dict["env_ql"] = np.array(scm_data_srs["env_ql"])[:,:]
    les_srs_dict["upd_ql"] = np.array(les_data_srs["updraft_ql"])[:,:]
    les_srs_dict["env_ql"] = np.array(les_data_srs["env_ql"])[:,:]

    scm_srs_dict["upd_ql_w"] = scm_srs_dict["upd_ql"] * scm_srs_dict["upd_area"]
    scm_srs_dict["env_ql_w"] = scm_srs_dict["env_ql"] * scm_srs_dict["env_area"]
    les_srs_dict["upd_ql_w"] = les_srs_dict["upd_ql"] * les_srs_dict["upd_area"]
    les_srs_dict["env_ql_w"] = les_srs_dict["env_ql"] * les_srs_dict["env_area"]

    # empty arrays for average quantities
    for key in ["mean_ql", "upd_ql", "upd_ql_w", "env_ql", "env_ql_w", "upd_cnt", "env_cnt"]:
        les_avg_dict[key] = np.zeros_like(height)
        scm_avg_dict[key] = np.zeros_like(height)

    # average only in places where you have updrafts/environment
    for it in range(-1, -1*(avg_step +1 ), -1):
        for iz in range(height.shape[0]):
            if scm_srs_dict["upd_area"][iz, it] > 0.:
                scm_avg_dict["upd_cnt"][iz]  += 1.
                for var in ["upd_ql", "upd_ql_w"]:
                    scm_avg_dict[var][iz] += scm_srs_dict[var][iz,it]
            if scm_srs_dict["env_area"][iz, it] > 0.:
                scm_avg_dict["env_cnt"][iz] += 1.
                for var in ["env_ql", "env_ql_w"]:
                    scm_avg_dict[var][iz] += scm_srs_dict[var][iz,it]
            if les_srs_dict["upd_area"][iz, it] > 0.:
                les_avg_dict["upd_cnt"][iz]  += 1.
                for var in ["upd_ql", "upd_ql_w"]:
                    les_avg_dict[var][iz] += les_srs_dict[var][iz,it]
            if les_srs_dict["env_area"][iz, it] > 0.:
                les_avg_dict["env_cnt"][iz]  += 1.
                for var in ["env_ql", "env_ql_w"]:
                    les_avg_dict[var][iz] += les_srs_dict[var][iz,it]

        scm_avg_dict["mean_ql"] += scm_srs_dict["mean_ql"][:,it]
        les_avg_dict["mean_ql"] += les_srs_dict["mean_ql"][:,it]

    for data_dict in [scm_avg_dict, les_avg_dict]:
        data_dict["mean_ql"] /= avg_step

    for iz in range(height.shape[0]):
        for var in ["upd_ql", "upd_ql_w"]:

            if scm_avg_dict["upd_cnt"][iz] > 0:
                scm_avg_dict[var][iz] /= scm_avg_dict["upd_cnt"][iz]
            else:
                scm_avg_dict[var][iz] = np.NaN

            if les_avg_dict["upd_cnt"][iz] > 0:
                les_avg_dict[var][iz] /= les_avg_dict["upd_cnt"][iz]
            else:
                les_avg_dict[var][iz] = np.NaN

        for var in ["env_ql", "env_ql_w"]:

            if scm_avg_dict["env_cnt"][iz] > 0:
                scm_avg_dict[var][iz] /= scm_avg_dict["env_cnt"][iz]
            else:
                scm_avg_dict[var][iz] = np.NaN

            if les_avg_dict["env_cnt"][iz] > 0:
                les_avg_dict[var][iz] /= les_avg_dict["env_cnt"][iz]
            else:
                les_avg_dict[var][iz] = np.NaN

    x_lab    = ["upd QL [g/kg]", "env QL [g/kg]", "mean QL [g/kg]", "upd QL * a [g/kg]", "env QL * a [g/kg]", "mean QL [g/kg]"]

    # iteration over plots
    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        if plot_it >= 3:
            plots[plot_it].set_xlim(x_range)
        plots[plot_it].set_ylim([0, height[-1] + (height[1] - height[0]) * 0.5])
        plots[plot_it].grid(True)
        for iz in focus_z:
            plots[plot_it].axhline(height[iz], c="peachpuff", lw=2)

    plots[0].plot(les_avg_dict["upd_ql"], height, ".-", color="blue")
    plots[0].plot(scm_avg_dict["upd_ql"], height, ".-", color="tomato")

    plots[1].plot(les_avg_dict["env_ql"], height, ".-", color="blue")
    plots[1].plot(scm_avg_dict["env_ql"], height, ".-", color="tomato")

    plots[2].plot(les_avg_dict["mean_ql"], height, ".-", color="blue",   label="les")
    plots[2].plot(scm_avg_dict["mean_ql"], height, ".-", color="tomato", label="offline scm")

    plots[3].plot(les_avg_dict["upd_ql_w"], height, ".-", color="blue")
    plots[3].plot(scm_avg_dict["upd_ql_w"], height, ".-", color="tomato")

    plots[4].plot(les_avg_dict["env_ql_w"], height, ".-", color="blue")
    plots[4].plot(scm_avg_dict["env_ql_w"], height, ".-", color="tomato")

    plots[5].plot(les_avg_dict["mean_ql"], height, ".-", color="blue")
    plots[5].plot(scm_avg_dict["mean_ql"], height, ".-", color="tomato")

    plots[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(simname)
    plt.savefig(folder + title)
    plt.clf()

def plot_percentiles(les_data_srs, scm_data_avg, title, simname, avg_step=30, folder="plots/output/"):
    """
    Plots LES QT and thetal with percentiles and covariance
    """
    suptitle = simname
    n_bins = 100
    # ranges to specify for plotting
    if simname == 'Bomex':
        focus_z  = [12, 27, 42, 62,  75, 90]
        y_range  = [0, 2000]

        th_range        = [298, 308]
        qt_range        = [4, 18]
        cov_range       = [-.6, .01]

        qt_spread_range = [-10, 10]
        th_spread_range = [-.25, .25]

        upd_dummy_qt_range = [11, 18]
        upd_dummy_th_range = [298.5, 302.5]
        env_dummy_qt_range = qt_range
        env_dummy_th_range = th_range

    elif simname == 'Dycoms_RF01':
        focus_z  = [120, 130, 140, 150,  160, 170]
        y_range  = [0, 1000]

        th_range        = [285, 305]
        qt_range        = [0, 11]
        cov_range       = [-1., .01]

        qt_spread_range = [-10, 10]
        th_spread_range = [-.25, .25]

        env_dummy_qt_range = qt_range
        env_dummy_th_range = th_range
        upd_dummy_qt_range = qt_range
        upd_dummy_th_range = th_range

    else:
        print "unrecognized simulation name"
        assert(False)

    # read data
    les_srs_dict = {}
    les_avg_dict = {}
    upd_var = ["updraft_qt", "updraft_thetali", "upd_QTvar", "upd_Hvar", "upd_HQTcov"]
    env_var = ["env_qt", "env_thetali", "env_QTvar", "env_Hvar", "env_HQTcov"]
    all_var = upd_var + env_var

    height = np.array(les_data_srs["z_half"][:])

    les_srs_dict["upd_area"] =  np.array(les_data_srs["updraft_fraction"])[:,:]
    les_srs_dict["env_area"] = 1. - les_srs_dict["upd_area"]
    for var in all_var:
        les_srs_dict[var] = np.array(les_data_srs[var][:,:])
        les_avg_dict[var] = np.zeros_like(height)
    for var in ["upd_cnt", "env_cnt"]:
        les_avg_dict[var] = np.zeros_like(height)

    for it in range(-1, -1*(avg_step +1 ), -1):
        for iz in range(height.shape[0]):
            if les_srs_dict["env_area"][iz, it] > 0:
                les_avg_dict["env_cnt"][iz] += 1
                for var in env_var:
                    les_avg_dict[var][iz] += les_srs_dict[var][iz,it]

            if les_srs_dict["upd_area"][iz, it] > 0:
                les_avg_dict["upd_cnt"][iz] += 1
                for var in upd_var:
                    les_avg_dict[var][iz] += les_srs_dict[var][iz,it]

    for var in env_var:
        for iz in range(height.shape[0]):
            if les_avg_dict["env_cnt"][iz] > 0:
                les_avg_dict[var][iz] /= les_avg_dict["env_cnt"][iz]
            else:
                les_avg_dict[var][iz] = np.NaN

    for var in upd_var:
        for iz in range(height.shape[0]):
            if les_avg_dict["upd_cnt"][iz] > 0:
                les_avg_dict[var][iz] /= les_avg_dict["upd_cnt"][iz]
            else:
                les_avg_dict[var][iz] = np.NaN

    # dictionaries to store percentiles
    upd_qt_perc = {}
    upd_th_perc = {}
    env_qt_perc = {}
    env_th_perc = {}
    for perc_dict in [upd_qt_perc, upd_th_perc, env_qt_perc, env_th_perc]:
        for perc in ["5", "45", "55", "95"]:
            perc_dict[perc] = np.zeros_like(height)

    # calculate percentiles
    for iz in range(height.shape[0]):
        # environment
        if les_avg_dict["env_QTvar"][iz] < 0:
            les_avg_dict["env_QTvar"][iz] = 0.
        qt_val = np.random.normal(les_avg_dict["env_qt"][iz], les_avg_dict["env_QTvar"][iz], 10000)
        if les_avg_dict["env_Hvar"][iz] < 0:
            les_avg_dict["env_Hvar"][iz] = 0.
        th_val = np.random.normal(les_avg_dict["env_thetali"][iz], les_avg_dict["env_Hvar"][iz], 10000)
        for perc in [5, 45, 55, 95]:
            env_qt_perc[str(perc)][iz] = np.percentile(qt_val, perc)
            env_th_perc[str(perc)][iz] = np.percentile(th_val, perc)
        # updraft
        qt_val = np.random.normal(les_avg_dict["updraft_qt"][iz], les_avg_dict["upd_QTvar"][iz], 10000)
        if les_avg_dict["upd_Hvar"][iz] < 0:
            les_avg_dict["upd_Hvar"][iz] = 0.
        th_val = np.random.normal(les_avg_dict["updraft_thetali"][iz], les_avg_dict["upd_Hvar"][iz], 10000)
        for perc in [5, 45, 55, 95]:
            upd_qt_perc[str(perc)][iz] = np.percentile(qt_val, perc)
            upd_th_perc[str(perc)][iz] = np.percentile(th_val, perc)

    # (1) plot profiles with percentiles
    fig = plt.figure(1)
    fig.set_figheight(10)
    fig.set_figwidth(12)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=3, markersize=6)

    x_lab    = ["env QT [g/kg]", "env TH [K]", "env cov [g/kg K]", "upd QT [g/kg]", "upd TH [K]", "upd cov [g/kg K]"]
    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim(y_range)
        plots[plot_it].grid(True)
        for iz in focus_z:
            plots[plot_it].axhline(height[iz], c="peachpuff", lw=2)

    plots[0].fill_betweenx(height, env_qt_perc["5"], env_qt_perc["95"], color="deepskyblue")
    plots[1].fill_betweenx(height, env_th_perc["5"], env_th_perc["95"], color="deepskyblue")
    plots[3].fill_betweenx(height, upd_qt_perc["5"], upd_qt_perc["95"], color="deepskyblue")
    plots[4].fill_betweenx(height, upd_th_perc["5"], upd_th_perc["95"], color="deepskyblue")

    plots[0].plot(les_avg_dict["env_qt"],          height, ".-", color="blue")
    plots[1].plot(les_avg_dict["env_thetali"],     height, ".-", color="blue")
    plots[2].plot(les_avg_dict["env_HQTcov"],      height, ".-", color="blue")
    plots[3].plot(les_avg_dict["updraft_qt"],      height, ".-", color="blue")
    plots[4].plot(les_avg_dict["updraft_thetali"], height, ".-", color="blue")
    plots[5].plot(les_avg_dict["upd_HQTcov"],      height, ".-", color="blue")

    for it in [0,3]:
        plots[it].set_xlim(qt_range)
    for it in [1,4]:
        plots[it].set_xlim(th_range)
    for it in [2,5]:
        plots[it].set_xlim(cov_range)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(suptitle)
    plt.savefig(folder + title)
    plt.clf()

    # (2) plot qt percentile spread
    fig = plt.figure(1)
    fig.set_figheight(10)
    fig.set_figwidth(12)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=3, markersize=6)

    plots = []
    for plot_it in range(4):
        plots.append(plt.subplot(2,2,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim(y_range)
        plots[plot_it].grid(True)
        for iz in focus_z:
            plots[plot_it].axhline(height[iz], c="peachpuff", lw=2)

    plots[0].plot(les_avg_dict["env_qt"],     height, ".-", color="blue",      label="env")
    plots[0].plot(les_avg_dict["updraft_qt"], height, ".-", color="lightblue", label="upd")

    plots[1].fill_betweenx(height, (env_qt_perc["5"]/les_avg_dict["env_qt"]  * 100 - 100), (env_qt_perc["95"]/les_avg_dict["env_qt"] * 100 - 100), color="deepskyblue")
    plots[1].fill_betweenx(height, (env_qt_perc["45"]/les_avg_dict["env_qt"] * 100 - 100), (env_qt_perc["55"]/les_avg_dict["env_qt"] * 100 - 100), color="blue")

    plots[2].plot(les_avg_dict["env_qt"],     height, ".-", color="lightblue", label="env")
    plots[2].plot(les_avg_dict["updraft_qt"], height, ".-", color="blue",      label="upd")

    plots[3].fill_betweenx(height, (upd_qt_perc["5"]/les_avg_dict["updraft_qt"]  * 100 - 100), (upd_qt_perc["95"]/les_avg_dict["updraft_qt"] * 100 - 100), color="deepskyblue")
    plots[3].fill_betweenx(height, (upd_qt_perc["45"]/les_avg_dict["updraft_qt"] * 100 - 100), (upd_qt_perc["55"]/les_avg_dict["updraft_qt"] * 100 - 100), color="blue")

    for it in [0,2]:
        plots[it].legend()
        plots[it].set_xlabel('QT [g/kg]')
        plots[it].set_xlim(qt_range)
    for it in [1,3]:
        plots[it].set_xlim(qt_spread_range)

    plots[1].set_xlabel('QT spread around env mean [%]')
    plots[3].set_xlabel('QT spread around upd mean [%]')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(suptitle)
    tmp_title = folder+title
    plt.savefig(tmp_title[0:-3] + "_spread_qt.pdf")
    plt.clf()

    # (3) plot th percentile spread
    fig = plt.figure(1)
    fig.set_figheight(10)
    fig.set_figwidth(12)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=3, markersize=6)

    plots = []
    for plot_it in range(4):
        plots.append(plt.subplot(2,2,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim(y_range)
        plots[plot_it].grid(True)
        for iz in focus_z:
            plots[plot_it].axhline(height[iz], c="peachpuff", lw=2)

    plots[0].plot(les_avg_dict["env_thetali"],     height, ".-", color="blue",      label="env")
    plots[0].plot(les_avg_dict["updraft_thetali"], height, ".-", color="lightblue", label="upd")

    plots[1].fill_betweenx(height, (env_th_perc["5"]/les_avg_dict["env_thetali"]  * 100 - 100), (env_th_perc["95"]/les_avg_dict["env_thetali"] * 100 - 100), color="deepskyblue")
    plots[1].fill_betweenx(height, (env_th_perc["45"]/les_avg_dict["env_thetali"] * 100 - 100), (env_th_perc["55"]/les_avg_dict["env_thetali"] * 100 - 100), color="blue")

    plots[2].plot(les_avg_dict["env_thetali"],     height, ".-", color="lightblue", label="env")
    plots[2].plot(les_avg_dict["updraft_thetali"], height, ".-", color="blue",      label="upd")

    plots[3].fill_betweenx(height, (upd_th_perc["5"]/les_avg_dict["updraft_thetali"]  * 100 - 100), (upd_th_perc["95"]/les_avg_dict["updraft_thetali"] * 100 - 100), color="deepskyblue")
    plots[3].fill_betweenx(height, (upd_th_perc["45"]/les_avg_dict["updraft_thetali"] * 100 - 100), (upd_th_perc["55"]/les_avg_dict["updraft_thetali"] * 100 - 100), color="blue")

    for it in [0,2]:
        plots[it].legend()
        plots[it].set_xlabel('TH [K]')
        plots[it].set_xlim(th_range)
    for it in [1,3]:
        plots[it].set_xlim(th_spread_range)

    plots[1].set_xlabel('TH spread around env mean [%]')
    plots[3].set_xlabel('TH spread around upd mean [%]')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(suptitle)
    tmp_title = folder+title
    plt.savefig(tmp_title[0:-3] + "_spread_th.pdf")
    plt.clf()

    # (4) plot 2D distributions of updrafts
    fig = plt.figure(1)
    fig.set_figheight(10)
    fig.set_figwidth(12)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=3, markersize=6)

    upd_dummy_qt = np.linspace(upd_dummy_qt_range[0], upd_dummy_qt_range[1], 1e3)
    upd_dummy_th = np.linspace(upd_dummy_th_range[0], upd_dummy_th_range[1], 1e3)

    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(3,2,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_title("z = " + str(height[focus_z[plot_it]]) + " [m]")
        plots[plot_it].set_xlabel("QT [g/kg]")
        plots[plot_it].set_ylabel("TH [K]")

        iz = focus_z[plot_it]
        upd_2d_values = np.vstack([\
            np.random.multivariate_normal(\
                [les_avg_dict["updraft_qt"][iz], les_avg_dict["updraft_thetali"][iz]],\
                [[les_avg_dict["upd_QTvar"][iz], les_avg_dict["upd_HQTcov"][iz]], [les_avg_dict["upd_HQTcov"][iz], les_avg_dict["upd_Hvar"][iz]]],\
                [100000]\
            )\
        ])

        hst = plots[plot_it].hist2d(\
                  upd_2d_values[:,0],\
                  upd_2d_values[:,1],\
                  bins=n_bins,\
                  range = [upd_dummy_qt_range, upd_dummy_th_range],\
                  cmap=cm.Blues,\
                  norm=LogNorm())
        plt.colorbar(hst[3], ax=plots[plot_it])

        upd_dummy_T       = upd_dummy_th * wrp.exner_c(scm_data_avg["p0_half"][focus_z[plot_it]])
        upd_dummy_qt_star = np.zeros_like(upd_dummy_T)
        for el_it in range(upd_dummy_T.shape[0]):
            upd_dummy_pv_star        = wrp.pv_star(upd_dummy_T[el_it])
            upd_dummy_qt_star[el_it] = wrp.qv_star_c(scm_data_avg["p0_half"][focus_z[plot_it]], upd_dummy_qt[el_it] * 1e-3, upd_dummy_pv_star) * 1e3

        plots[plot_it].plot(upd_dummy_qt_star, upd_dummy_th, '-', lw=1, c='red')

        plots[plot_it].axhline(les_avg_dict["updraft_thetali"][iz], c="lightgray", lw=1)
        plots[plot_it].axvline(les_avg_dict["updraft_qt"][iz],      c="lightgray", lw=1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(suptitle + " - 2D updraft histograms")
    tmp_title = folder+title
    plt.savefig(tmp_title[0:-3] + "_2d_hist_upd.pdf")
    plt.clf()

    # (5) plot 2D distributions of environment
    fig = plt.figure(1)
    fig.set_figheight(10)
    fig.set_figwidth(12)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=3, markersize=6)

    env_dummy_qt = np.linspace(env_dummy_qt_range[0], env_dummy_qt_range[1], 1e3)
    env_dummy_th = np.linspace(env_dummy_th_range[0], env_dummy_th_range[1], 1e3)

    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(3,2,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_title("z = " + str(height[focus_z[plot_it]]) + " [m]")
        plots[plot_it].set_xlabel("QT [g/kg]")
        plots[plot_it].set_ylabel("TH [K]")

        iz = focus_z[plot_it]
        env_2d_values = np.vstack([\
            np.random.multivariate_normal(\
                [les_avg_dict["env_qt"][iz], les_avg_dict["env_thetali"][iz]],\
                [[les_avg_dict["env_QTvar"][iz], les_avg_dict["env_HQTcov"][iz]], [les_avg_dict["env_HQTcov"][iz], les_avg_dict["env_Hvar"][iz]]],\
                [100000]\
            )\
        ])

        hst = plots[plot_it].hist2d(\
                  env_2d_values[:,0],\
                  env_2d_values[:,1],\
                  bins=n_bins,\
                  range = [env_dummy_qt_range, env_dummy_th_range],\
                  cmap=cm.Blues,\
                  norm=LogNorm())
        plt.colorbar(hst[3], ax=plots[plot_it])

        env_dummy_T       = env_dummy_th * wrp.exner_c(scm_data_avg["p0_half"][focus_z[plot_it]])
        env_dummy_qt_star = np.zeros_like(env_dummy_T)
        for el_it in range(env_dummy_T.shape[0]):
            env_dummy_pv_star        = wrp.pv_star(env_dummy_T[el_it])
            env_dummy_qt_star[el_it] = wrp.qv_star_c(scm_data_avg["p0_half"][focus_z[plot_it]], env_dummy_qt[el_it] * 1e-3, env_dummy_pv_star) * 1e3

        plots[plot_it].plot(env_dummy_qt_star, env_dummy_th, '-', lw=1, c='red')

        plots[plot_it].axhline(les_avg_dict["env_thetali"][iz], c="lightgray", lw=1)
        plots[plot_it].axvline(les_avg_dict["env_qt"][iz],      c="lightgray", lw=1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(suptitle + " - 2D environment histograms")
    tmp_title = folder+title
    plt.savefig(tmp_title[0:-3] + "_2d_hist_env.pdf")
    plt.clf()


def plot_var_covar_mean(data, title, folder="plots/output/"):
    """
    Plots variance and covariance profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 16})
    mpl.rc('lines', linewidth=4, markersize=10)

    # data to plot
    x_lab         = ["Hvar",       "QTvar",       "HQTcov"]
    plot_var_mean = ["Hvar_mean",  "QTvar_mean",  "HQTcov_mean"]
    plot_var_env  = ["env_Hvar",   "env_QTvar",   "env_HQTcov"]

    # iteration over plots
    plots = []
    for plot_it in range(3):
        plots.append(plt.subplot(1,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        plots[plot_it].xaxis.set_major_locator(ticker.MaxNLocator(2))

        plots[plot_it].plot(data[plot_var_mean[plot_it]][-1], data["z_half"], ".-", label=plot_var_mean[plot_it], c="black")
        plots[plot_it].plot(data[plot_var_env[plot_it]][-1],  data["z_half"], ".-", label=plot_var_env[plot_it],  c="red")

    plots[0].legend()
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_var_covar_components(data, title, folder="plots/output/"):
    """
    Plots variance and covariance components profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 16})
    mpl.rc('lines', linewidth=4, markersize=10)

    # data to plot
    plot_Hvar_c   = ["Hvar_dissipation",   "Hvar_entr_gain",   "Hvar_detr_loss",   "Hvar_shear",   "Hvar_rain"]
    plot_QTvar_c  = ["QTvar_dissipation",  "QTvar_entr_gain",  "QTvar_detr_loss",  "QTvar_shear",  "QTvar_rain"]
    plot_HQTcov_c = ["HQTcov_dissipation", "HQTcov_entr_gain", "HQTcov_detr_loss", "HQTcov_shear", "HQTcov_rain"]
    color_c       = ['green',              'pink',             'purple',           'orange',       'blue']

    x_lab         = ["Hvar",      "QTvar",      "HQTcov"]
    plot_var_data = [plot_Hvar_c, plot_QTvar_c, plot_HQTcov_c]

    # iteration over plots
    plots = []
    for plot_it in range(3):
        plots.append(plt.subplot(1,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        plots[plot_it].xaxis.set_major_locator(ticker.MaxNLocator(2))

        for var in range(5):
            plots[plot_it].plot(data[plot_var_data[plot_it][var]][-1],   data["z_half"], ".-", label=plot_Hvar_c[var],  c=color_c[var])

    plots[0].legend()
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_timeseries_1D(data, title, folder="plots/output/"):
    """
    Plots timeseries from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=2, markersize=6)

    # data to plot
    plot_y = [data["updraft_cloud_cover"], data["updraft_cloud_top"], data["updraft_lwp"], data["ustar"], data["lhf"], data["Tsurface"]]
    y_lab  = ['updr cl. cover',            'updr CB, CT',             'LWP',       'u star',      'shf',       'T surf']

    # iteration over plots
    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel('t [s]')
        plots[plot_it].set_ylabel(y_lab[plot_it])
        plots[plot_it].set_xlim([0, data["t"][-1]])
        plots[plot_it].grid(True)
        plots[plot_it].plot(data["t"], plot_y[plot_it], '.-', color="b")
        if plot_it == 1:
            plots[plot_it].plot(data["t"], data["updraft_cloud_base"], '.-', color="r", label="CB")
            plots[plot_it].plot(data["t"], data["updraft_cloud_top"],  '.-', color="b", label="CT")
            plots[plot_it].legend()
        if plot_it == 4:
            plots[plot_it].plot(data["t"], data["lhf"], '.-', color="r", label='lhf')
            plots[plot_it].plot(data["t"], data["shf"], '.-', color="b", label='shf')
            plots[plot_it].legend()

    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_timeseries(data, case, folder="plots/output/"):
    """
    Plots the time series of Scampy simulations

    Input:
    data   - dictionary with previousely read it data
    case   - name for the tested simulation (to be used in the plot name)
    folder - folder where to save the created plot
    """
    # customize figure parameters
    fig = plt.figure(1)
    fig.set_figheight(15)
    fig.set_figwidth(30)
    mpl.rcParams.update({'font.size': 20})

    # read data
    z_half   = data["z_half"]
    time     = data["t"] / 60. / 60.
    mean_qv  = data["qt_mean"]    - data["ql_mean"]
    updr_qv  = data["updraft_qt"] - data["updraft_ql"]
    env_qv   = data["env_qt"]     - data["env_ql"]
    env_area = 1. - data["updraft_area"]

    # data to plot
    #mean_data  = [data["thetal_mean"],    data["buoyancy_mean"],    data["tke_mean"],      mean_qv,                data["ql_mean"],           data["qr_mean"]]
    #mean_label = ["mean thl [K]",         "mean buo [cm2/s3]",      "mean TKE [m2/s2]",    "mean qv [g/kg]",       "mean ql [g/kg]",          "mean qr [g/kg]"]
    #mean_cb    = [mpl.cm.Reds,            mpl.cm.Reds,              mpl.cm.Reds,           mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]
    #mean_min   = [299,                    -10,                       0,                    2,                      0,                          0]
    #mean_max   = [316,                    10,                        0.5,                  16,                     16e-5,                     1]

    env_data   = [data["env_thetal"],     env_area,                 data["env_w"],         env_qv,                 data["env_ql"],            data["env_qr"]]
    env_label  = ["env thl [K]",          "env area [%]",           "env w [m/s]",         "env qv [g/kg]",        "env ql [g/kg]",           "env qr [g/kg]"]
    env_cb     = [mpl.cm.Reds,            mpl.cm.Reds,              mpl.cm.Reds_r,         mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]
    env_min    = [299,                    .9,                       -0.05,                 2,                      0,                         0]
    env_max    = [316,                    1,                        0,                     16,                     12e-16,                    1]

    updr_data  = [data["updraft_thetal"], data["updraft_area"],     data["updraft_w"],     updr_qv,                data["updraft_ql"],        data["updraft_qr"]]
    updr_label = ["updr thl [K]",   "     updr area [%]",           "updr w [m/s]",        "updr qv [g/kg]",       "updr ql [g/kg]",          "updr qr [g/kg"]
    updr_cb    = [mpl.cm.Reds,            mpl.cm.Reds,              mpl.cm.Reds,           mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]
    updr_min   = [299,                    0,                        0,                     2,                      0,                         0]
    updr_max   = [316,                    0.1,                      0.025,                 16,                     14e-4,                     1]

    #flux_data  = [data["massflux_h"],     data["diffusive_flux_h"], data["total_flux_h"],  data["massflux_qt"],    data["diffusive_flux_qt"], data["total_flux_qt"]]
    #flux_label = ["M_FL thl",             "D_FL thl ",              "tot FL thl",          "M_FL qt",              "D_FL qt",                 "tot FL qt"]
    #flux_cb    = [mpl.cm.Spectral,        mpl.cm.Spectral,          mpl.cm.Spectral,       mpl.cm.Spectral_r,      mpl.cm.Spectral_r,         mpl.cm.Spectral_r]
    #flux_min   = [-12e-3,                 -40e-3,                   -0.04,                 0,                      0,                         0]
    #flux_max   = [16e-3,                  5e-3,                     0.01,                  0.08,                   0.07,                      0.1]

    #misc_data  = [data["eddy_viscosity"], data["eddy_diffusivity"], data["mixing_length"], data["entrainment_sc"], data["detrainment_sc"],    data["massflux"]]
    #misc_label = ["eddy visc",            "eddy diff",              "mix. length",         "entr sc",              "detr sc",                 "mass flux"]
    #misc_cb    = [mpl.cm.Blues,           mpl.cm.Blues,             mpl.cm.Blues,          mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]
    #misc_min   = [0,                      0,                        0,                     0.,                     0,                         0]
    #misc_max   = [20,                     20,                       350,                   0.06,                   0.14,                      0.05]

    rain_data  = [data["updraft_area"],   data["updraft_rain_area"],data["updraft_qr"],    env_area,               data["env_rain_area"],     data["env_qr"]]
    rain_label = ["updr area",            "updr rain area",         "updr qr",             "env area",             "env rain area",           "env rain"]
    rain_cb    = [mpl.cm.Spectral,        mpl.cm.Spectral,          mpl.cm.Spectral,       mpl.cm.Spectral,        mpl.cm.Spectral,           mpl.cm.Spectral]
    rain_min   = [0,                      0,                        0,                     .9,                     0,                         0]
    rain_max   = [0.1,                    0.3,                      1,                     1,                      0.2,                       1]

    data_to_plot = [env_data,  updr_data,  rain_data]
    labels       = [env_label, updr_label, rain_label ]
    titles       = ["02env",   "03updr",   "06rain"]
    cbs          = [env_cb,    updr_cb,    rain_cb]
    vmin         = [env_min,   updr_min,   rain_min]
    vmax         = [env_max,   updr_max,   rain_max]

    # iteration over plots
    for var in range(3):
        ax   = []
        plot = []
        for plot_it in range(6):
            ax.append(fig.add_subplot(2,3,plot_it+1))
                                    #(rows, columns, number)
            ax[plot_it].set_xlabel('t [hrs]')
            ax[plot_it].set_ylabel('z [m]')
            plot.append(ax[plot_it].pcolormesh(time[:], z_half, data_to_plot[var][plot_it][:,:], cmap=discrete_cmap(32, cbs[var][plot_it]), rasterized=True))
            #plot.append(ax[plot_it].pcolormesh(time[-9:], z_half, data_to_plot[var][plot_it][:,-9:], cmap=discrete_cmap(32, cbs[var][plot_it]), vmin=vmin[var][plot_it], vmax=vmax[var][plot_it], rasterized=True))
            fig.colorbar(plot[plot_it], ax=ax[plot_it], label=labels[var][plot_it])

        #plt.tight_layout()
        plt.savefig(folder + case + "_timeseries_" + titles[var] + ".pdf")
        plt.clf()
