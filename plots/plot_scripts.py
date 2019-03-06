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

def read_LES_data_avg(sim_data, n_steps=0):
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
    data_to_plot["z_half"]  = np.array(sim_data["reference/z_half"][:])
    data_to_plot["p0_half"] = np.array(sim_data["reference/p0_half"][:])

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

def plot_drafts_area_wght(scm_data_avg, scm_data_srs, les_data_avg, les_data_srs, title, folder="plots/output/"):
    """
    Plots updraft and environment area weighted profiles from Scampy and LES data

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=3, markersize=6)

    # read data
    tmp = np.array(scm_data_avg["ql_mean"])[-1]
    scm_env_area = 1. - np.array(scm_data_srs["updraft_area"])[:,:]
    les_env_area = 1. - np.array(les_data_srs["updraft_fraction"])[:,:]

    scm_mean_ql_srs = np.array(scm_data_srs["ql_mean"])[:,:]
    les_mean_ql_srs = np.array(les_data_srs["ql_mean"])[:,:]

    scm_upd_ql_srs = np.array(scm_data_srs["updraft_ql"])[:,:] * np.array(scm_data_srs["updraft_area"])[:,:]
    scm_env_ql_srs = np.array(scm_data_srs["env_ql"])[:,:]     * scm_env_area
    les_upd_ql_srs = np.array(les_data_srs["updraft_ql"])[:,:] * np.array(les_data_srs["updraft_fraction"])[:,:]
    les_env_ql_srs = np.array(les_data_srs["env_ql"])[:,:]     * les_env_area

    scm_upd_ql = np.zeros_like(tmp)
    les_upd_ql = np.zeros_like(tmp)

    scm_env_ql = np.zeros_like(tmp)
    les_env_ql = np.zeros_like(tmp)

    scm_mean_ql = np.zeros_like(tmp)
    les_mean_ql = np.zeros_like(tmp)

    avg_step = 50
    for it in range(-1, -1*(avg_step +1 ), -1):
        scm_upd_ql  += scm_upd_ql_srs[:,it]
        scm_env_ql  += scm_env_ql_srs[:,it]
        les_upd_ql  += les_upd_ql_srs[:,it]
        les_env_ql  += les_env_ql_srs[:,it]
        scm_mean_ql += scm_mean_ql_srs[:,it]
        les_mean_ql += les_mean_ql_srs[:,it]

    for arr in [scm_upd_ql, scm_env_ql, les_upd_ql, les_env_ql, scm_mean_ql, les_mean_ql]:
        arr /= avg_step

    x_lab    = ["upd QL [g/kg]", "env QL [g/kg]", "mean QL [g/kg]"]
    focus_z = [12, 30, 42, 62,  75, 87]

    # iteration over plots
    plots = []
    for plot_it in range(3):
        plots.append(plt.subplot(1,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_xlim([-.001, .02])
        plots[plot_it].set_ylim([0, scm_data_avg["z_half"][-1] + (scm_data_avg["z_half"][1] - scm_data_avg["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        for iz in focus_z:
            plots[plot_it].axhline(scm_data_avg["z_half"][iz], c="peachpuff", lw=2)

    plots[0].plot(scm_upd_ql, scm_data_avg["z_half"], ".-", color="tomato")
    plots[0].plot(les_upd_ql, les_data_avg["z_half"], ".-", color="blue")

    plots[1].plot(scm_env_ql, scm_data_avg["z_half"], ".-", color="tomato")
    plots[1].plot(les_env_ql, les_data_avg["z_half"], ".-", color="blue")

    plots[2].plot(scm_mean_ql, scm_data_avg["z_half"], ".-", color="tomato", label="offline scm")
    plots[2].plot(les_mean_ql, les_data_avg["z_half"], ".-", color="blue",   label="les")

    plots[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Bomex")
    plt.savefig(folder + title)
    plt.clf()


def plot_percentiles(les_data_srs, scm_data_avg, title, folder="plots/output/"):
    """
    Plots LES QT and thetal with percentiles and covariance
    """
    # read data
    upd_area = les_data_srs["updraft_fraction"][:,:]
    env_area = 1. - upd_area

    qt_env_srs = les_data_srs["env_qt"][:,:]
    th_env_srs = les_data_srs["env_thetali"][:,:]
    qt_var_env_srs = les_data_srs["env_QTvar"][:,:]
    th_var_env_srs = les_data_srs["env_Hvar"][:,:]
    cov_env_srs = les_data_srs["env_HQTcov"][:,:]

    qt_upd_srs = les_data_srs["updraft_qt"][:,:]
    th_upd_srs = les_data_srs["updraft_thetali"][:,:]
    qt_var_upd_srs = les_data_srs["upd_QTvar"][:,:]
    th_var_upd_srs = les_data_srs["upd_Hvar"][:,:]
    cov_upd_srs = les_data_srs["upd_HQTcov"][:,:]

    avg_step = 25
    tmp = np.array(scm_data_avg["ql_mean"])[-1]

    qt_env = np.zeros_like(tmp)
    th_env = np.zeros_like(tmp)
    qt_upd = np.zeros_like(tmp)
    th_upd = np.zeros_like(tmp)
    qt_var_env = np.zeros_like(tmp)
    th_var_env = np.zeros_like(tmp)
    qt_var_upd = np.zeros_like(tmp)
    th_var_upd = np.zeros_like(tmp)
    cov_env = np.zeros_like(tmp)
    cov_upd = np.zeros_like(tmp)

    tmp_upd_cnt = np.zeros_like(tmp)
    tmp_env_cnt = np.zeros_like(tmp)

    for it in range(-1, -1*(avg_step +1 ), -1):
        for iz in range(tmp.shape[0]):
            if env_area[iz, it] > 0:
                qt_env[iz]  += qt_env_srs[iz,it]
                th_env[iz]  += th_env_srs[iz,it]
                qt_var_env[iz] += qt_var_env_srs[iz,it]
                th_var_env[iz] += th_var_env_srs[iz,it]
                cov_env[iz] += cov_env_srs[iz,it]
                tmp_env_cnt[iz] += 1

            if upd_area[iz, it] > 0:
                qt_upd[iz]  += qt_upd_srs[iz,it]
                th_upd[iz]  += th_upd_srs[iz,it]
                qt_var_upd[iz] += qt_var_upd_srs[iz,it]
                th_var_upd[iz] += th_var_upd_srs[iz,it]
                cov_upd[iz] += cov_upd_srs[iz,it]
                tmp_upd_cnt[iz] += 1

    for arr in [qt_env, th_env, qt_var_env, th_var_env, cov_env]:
        for iz in range(arr.shape[0]):
            if tmp_env_cnt[iz] > 0:
                arr[iz] /= tmp_env_cnt[iz]
            else:
                arr[iz] = np.NaN

    for arr in [qt_upd, th_upd, qt_var_upd, th_var_upd, cov_upd]:
        for iz in range(arr.shape[0]):
            if tmp_upd_cnt[iz] > 0:
                arr[iz] /= tmp_upd_cnt[iz]
            else:
                arr[iz] = np.NaN

    qt_05 = np.zeros_like(tmp)
    qt_45 = np.zeros_like(tmp)
    qt_55 = np.zeros_like(tmp)
    qt_95 = np.zeros_like(tmp)

    th_05 = np.zeros_like(tmp)
    th_45 = np.zeros_like(tmp)
    th_55 = np.zeros_like(tmp)
    th_95 = np.zeros_like(tmp)

    qt_u_05 = np.zeros_like(tmp)
    qt_u_45 = np.zeros_like(tmp)
    qt_u_55 = np.zeros_like(tmp)
    qt_u_95 = np.zeros_like(tmp)

    th_u_05 = np.zeros_like(tmp)
    th_u_45 = np.zeros_like(tmp)
    th_u_55 = np.zeros_like(tmp)
    th_u_95 = np.zeros_like(tmp)

    # calculate percentiles
    for iz in range(tmp.shape[0]):
        qt_val = np.random.normal(qt_env[iz], qt_var_env[iz], 10000)
        qt_05[iz] = np.percentile(qt_val, 5)
        qt_45[iz] = np.percentile(qt_val, 45)
        qt_55[iz] = np.percentile(qt_val, 55)
        qt_95[iz] = np.percentile(qt_val, 95)

        if th_var_env[iz] < 0:
            th_var_env[iz] = 0.
        th_val = np.random.normal(th_env[iz], th_var_env[iz], 10000)
        th_05[iz] = np.percentile(th_val, 5)
        th_45[iz] = np.percentile(th_val, 45)
        th_55[iz] = np.percentile(th_val, 55)
        th_95[iz] = np.percentile(th_val, 95)

    # calculate percentiles
    for iz in range(tmp.shape[0]):
        qt_u_val = np.random.normal(qt_upd[iz], qt_var_upd[iz], 10000)
        qt_u_05[iz] = np.percentile(qt_u_val, 5)
        qt_u_45[iz] = np.percentile(qt_u_val, 45)
        qt_u_55[iz] = np.percentile(qt_u_val, 55)
        qt_u_95[iz] = np.percentile(qt_u_val, 95)

        th_u_val = np.random.normal(th_upd[iz], th_var_upd[iz], 10000)
        th_u_05[iz] = np.percentile(th_u_val, 5)
        th_u_45[iz] = np.percentile(th_u_val, 45)
        th_u_55[iz] = np.percentile(th_u_val, 55)
        th_u_95[iz] = np.percentile(th_u_val, 95)


    focus_z = [12, 30, 42, 62,  75, 87]

    # iteration over plots
    # customize defaults
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
        plots[plot_it].set_ylim([0, 2000])
        plots[plot_it].grid(True)
        for iz in focus_z:
            plots[plot_it].axhline(scm_data_avg["z_half"][iz], c="peachpuff", lw=2)

    plots[0].fill_betweenx(scm_data_avg["z_half"], qt_05, qt_95, color="deepskyblue")
    plots[1].fill_betweenx(scm_data_avg["z_half"], th_05, th_95, color="deepskyblue")
    plots[3].fill_betweenx(scm_data_avg["z_half"], qt_u_05, qt_u_95, color="deepskyblue")
    plots[4].fill_betweenx(scm_data_avg["z_half"], th_u_05, th_u_95, color="deepskyblue")

    plots[0].plot(qt_env,  scm_data_avg["z_half"], ".-", color="blue")
    plots[1].plot(th_env,  scm_data_avg["z_half"], ".-", color="blue")
    plots[2].plot(cov_env, scm_data_avg["z_half"], ".-", color="blue")

    plots[3].plot(qt_upd,  scm_data_avg["z_half"], ".-", color="blue")
    plots[4].plot(th_upd,  scm_data_avg["z_half"], ".-", color="blue")
    plots[5].plot(cov_upd, scm_data_avg["z_half"], ".-", color="blue")

    plots[0].set_xlim([5, 18])
    plots[3].set_xlim([5, 18])
    plots[1].set_xlim([298, 308])
    plots[4].set_xlim([298, 308])
    plots[2].set_xlim([-0.6, 0.01])
    plots[5].set_xlim([-0.6, 0.01])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Bomex")
    plt.savefig(folder + title)
    plt.clf()

    # iteration over plots
    # customize defaults
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
        plots[plot_it].set_ylim([0, 2000])
        plots[plot_it].grid(True)
        for iz in focus_z:
            plots[plot_it].axhline(scm_data_avg["z_half"][iz], c="peachpuff", lw=2)


    plots[0].plot(qt_env,  scm_data_avg["z_half"], ".-", color="blue", label="env")
    plots[0].plot(qt_upd,  scm_data_avg["z_half"], ".-", color="lightblue", label="upd")

    plots[1].fill_betweenx(scm_data_avg["z_half"], (qt_05/qt_env * 100 - 100), (qt_95/qt_env * 100 - 100), color="deepskyblue")
    plots[1].fill_betweenx(scm_data_avg["z_half"], (qt_45/qt_env * 100 - 100), (qt_55/qt_env * 100 - 100), color="blue")

    plots[2].plot(qt_env,  scm_data_avg["z_half"], ".-", color="lightblue", label="env")
    plots[2].plot(qt_upd,  scm_data_avg["z_half"], ".-", color="blue", label="upd")

    plots[3].fill_betweenx(scm_data_avg["z_half"], (qt_u_05/qt_upd * 100 - 100), (qt_u_95/qt_upd * 100 - 100), color="deepskyblue")
    plots[3].fill_betweenx(scm_data_avg["z_half"], (qt_u_45/qt_upd * 100 - 100), (qt_u_55/qt_upd * 100 - 100), color="blue")

    plots[0].legend()
    plots[2].legend()
    plots[0].set_xlabel('QT [g/kg]')
    plots[2].set_xlabel('QT [g/kg]')
    plots[1].set_xlabel('QT spread around env mean [%]')
    plots[3].set_xlabel('QT spread around upd mean [%]')

    plots[1].set_xlim([-10, 10])
    plots[3].set_xlim([-10, 10])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Bomex")
    tmp_title = folder+title
    plt.savefig(tmp_title[0:-3] + "_spread_qt.pdf")
    plt.clf()

    # iteration over plots
    # customize defaults
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
        plots[plot_it].set_ylim([0, 2000])
        plots[plot_it].grid(True)
        for iz in focus_z:
            plots[plot_it].axhline(scm_data_avg["z_half"][iz], c="peachpuff", lw=2)

    plots[0].plot(th_env,  scm_data_avg["z_half"], ".-", color="blue", label="env")
    plots[0].plot(th_upd,  scm_data_avg["z_half"], ".-", color="lightblue", label="upd")

    plots[1].fill_betweenx(scm_data_avg["z_half"], (th_05/th_env * 100 - 100), (th_95/th_env * 100 - 100), color="deepskyblue")
    plots[1].fill_betweenx(scm_data_avg["z_half"], (th_45/th_env * 100 - 100), (th_55/th_env * 100 - 100), color="blue")

    plots[2].plot(th_env,  scm_data_avg["z_half"], ".-", color="lightblue", label="env")
    plots[2].plot(th_upd,  scm_data_avg["z_half"], ".-", color="blue", label="upd")

    plots[3].fill_betweenx(scm_data_avg["z_half"], (th_u_05/th_upd * 100 - 100), (th_u_95/th_upd * 100 - 100), color="deepskyblue")
    plots[3].fill_betweenx(scm_data_avg["z_half"], (th_u_45/th_upd * 100 - 100), (th_u_55/th_upd * 100 - 100), color="blue")

    plots[0].legend()
    plots[2].legend()
    plots[0].set_xlabel('TH [K]')
    plots[2].set_xlabel('TH [K]')
    plots[1].set_xlabel('TH spread around env mean [%]')
    plots[3].set_xlabel('TH spread around upd mean [%]')

    plots[0].set_xlim([298, 308])
    plots[2].set_xlim([298, 308])

    plots[1].set_xlim([-.25, .25])
    plots[3].set_xlim([-.25, .25])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Bomex")
    tmp_title = folder+title
    plt.savefig(tmp_title[0:-3] + "_spread_th.pdf")
    plt.clf()

    # 2D distributions
    fig = plt.figure(1)
    fig.set_figheight(10)
    fig.set_figwidth(12)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=3, markersize=6)

    dummy_qt = np.arange(11,18,.1)
    dummy_th = np.arange(298.5, 302.5, .1)
    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(3,2,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_title("z = " + str(scm_data_avg["z_half"][focus_z[plot_it]]) + " [m]")
        plots[plot_it].set_xlabel("QT [g/kg]")
        plots[plot_it].set_ylabel("TH [K]")

        iz = focus_z[plot_it]
        upd_2d_values = np.vstack([\
            np.random.multivariate_normal(\
                [qt_upd[iz], th_upd[iz]],\
                [[qt_var_upd[iz], cov_upd[iz]], [cov_upd[iz], th_var_upd[iz]]],\
                [10000]\
            )\
        ])

        hst = plots[plot_it].hist2d(\
                  upd_2d_values[:,0],\
                  upd_2d_values[:,1],\
                  bins=100,\
                  range = [[11,18], [298.5, 302.5]],\
                  cmap=cm.Blues)
        plt.colorbar(hst[3], ax=plots[plot_it])

        #TODO
        # plot the same as weighted ql but with just ql
        # plot actual histograms from les

        dummy_T  = dummy_th * wrp.exner_c(scm_data_avg["p0_half"][focus_z[plot_it]])
        dummy_qt_star = np.zeros_like(dummy_T)
        for el_it in range(dummy_T.shape[0]):
            dummy_pv_star = wrp.pv_star(dummy_T[el_it])
            dummy_qt_star[el_it] = wrp.qv_star_c(scm_data_avg["p0_half"][focus_z[plot_it]], dummy_qt[el_it] * 1e-3, dummy_pv_star) * 1e3

        plots[plot_it].plot(dummy_qt_star, dummy_th, '-', lw=2, c='red')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Bomex - 2D updraft histograms")
    tmp_title = folder+title
    plt.savefig(tmp_title[0:-3] + "_2d_hist_upd.pdf")
    plt.clf()

    # 2D distributions
    fig = plt.figure(1)
    fig.set_figheight(10)
    fig.set_figwidth(12)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=3, markersize=6)

    dummy_qt = np.arange(4,18,.1)
    dummy_th = np.arange(298.5, 308.5, .1)
    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(3,2,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_title("z = " + str(scm_data_avg["z_half"][focus_z[plot_it]]) + " [m]")
        plots[plot_it].set_xlabel("QT [g/kg]")
        plots[plot_it].set_ylabel("TH [K]")

        iz = focus_z[plot_it]
        env_2d_values = np.vstack([\
            np.random.multivariate_normal(\
                [qt_env[iz], th_env[iz]],\
                [[qt_var_env[iz], cov_env[iz]], [cov_env[iz], th_var_env[iz]]],\
                [10000]\
            )\
        ])

        hst = plots[plot_it].hist2d(\
                  env_2d_values[:,0],\
                  env_2d_values[:,1],\
                  bins=100,\
                  range = [[4,18], [298.5, 308.5]],\
                  cmap=cm.Blues)
        plt.colorbar(hst[3], ax=plots[plot_it])

        dummy_T  = dummy_th * wrp.exner_c(scm_data_avg["p0_half"][focus_z[plot_it]])
        dummy_qt_star = np.zeros_like(dummy_T)
        for el_it in range(dummy_T.shape[0]):
            dummy_pv_star = wrp.pv_star(dummy_T[el_it])
            dummy_qt_star[el_it] = wrp.qv_star_c(scm_data_avg["p0_half"][focus_z[plot_it]], dummy_qt[el_it] * 1e-3, dummy_pv_star) * 1e3

        plots[plot_it].plot(dummy_qt_star, dummy_th, '-', lw=2, c='red')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Bomex - 2D environment histograms")
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
