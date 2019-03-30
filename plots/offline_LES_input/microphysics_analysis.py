import numpy as np
import netCDF4 as nc
import scipy.stats as stats
import math as mt

import json as simplejson

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import pylab as pl

import sys
sys.path.insert(0, "../../../SCAMPY/")
import scampy_wrapper as wrp

from parameters import *
from masking import *

#sim_name = 'DYCOMS_RF01'
sim_name = 'Bomex'

show_fractions = True
n_bins = 100
if sim_name == 'Bomex':
    path_to_data  = '../BOMEX_comparison/Bomex_SA_LES_data/'
    LES_data_file = path_to_data + 'fields/1021600.nc'
    refrence_file = path_to_data + 'Stats.Bomex.nc'
    # height indices from which I plot 2D histograms
    focus_levels  = [12, 27, 42, 62,  75, 85]
    y_range = [0, 2000]
    # th and qt limits for percentile plots
    th_range  = [298, 308]
    qt_range  = [4, 18]

    qt_spread = [-10, 10]
    th_spread = [-.25, .25]

    upd_qt_range = [11,18]
    upd_th_range = [298.5, 302.5]
    env_qt_range = qt_range
    env_th_range = th_range

    z_half_flag = True

elif sim_name == 'DYCOMS_RF01':
    path_to_data  = '../DYCOMS_comparison/DYCOMS_SA_LES_data_new/'
    LES_data_file = path_to_data + 'fields/14400.nc'
    refrence_file = path_to_data + 'Stats.DYCOMS_RF01.nc'
    # height indices from which I plot 2D histograms
    focus_levels  = [120, 130, 140, 150, 160, 170]
    y_range = [0, 1000]
    # th and qt limits for percentile plots
    th_range  = [285, 305]
    qt_range  = [0, 11]

    qt_spread = [-10, 10]
    th_spread = [-.25, .25]

    env_qt_range = qt_range
    env_th_range = th_range
    upd_qt_range = qt_range
    upd_th_range = th_range

    z_half_flag = False

elif sim_name == 'RICO':
    file_folder   = '/export/data1/ajaruga/data/RICO/e9a09/'
    ncdf_folder   = file_folder + 'fields/'
    time          = '19800'
    tmp_file      = ncdf_folder + time + '.nc'

else:
    print "unrecognised simulation name"
    assert(False)

# open LES netcdf files
# (The LES output doesnt have all the reference profiles that I need.
#  I'm taking them from SCAMPy otput files.
#  This only works if my vertical resolution is the same in LES and SCM)
print "reading data from file: ",          LES_data_file
print "reading reference profiles from: ", refrence_file
root     = nc.Dataset(LES_data_file, 'r')
ref_root = nc.Dataset(refrence_file, 'r')

ql     = root.groups['fields'].variables['ql'][:, :, :]
w      = root.groups['fields'].variables['w'][:, :, :]
tracer = root.groups['fields'].variables['c_srf_15'][:,:,:]
th     = root.groups['fields'].variables['thetali'][:, :, :]
qt     = root.groups['fields'].variables['qt'][:, :, :]

dims = qt.shape
nx = dims[0]
ny = dims[1]
nz = dims[2]

if z_half_flag:
   z_half = ref_root.groups['reference'].variables['z_half'][:]
   p0_half = ref_root.groups['reference'].variables['p0_half'][:]
else:
   p0_half = ref_root.groups['reference'].variables['p0'][:]
   z_half = ref_root.groups['reference'].variables['z'][:]
height  = z_half[0 : -1]

# reshape variables
ql_2d       = reshape_var(ql)
w_2d        = reshape_var(w)
tracer_2d   = reshape_var(tracer)
th_2d       = reshape_var(th)
qt_2d       = reshape_var(qt)
# get cloud layer
cb, ct      = calculate_cloud_base_top(ql_2d)
# interpolate vertical velocity
w_interp_2d = interpolate_w(w_2d)
#calculate masks
mask_dict = updraft_env_mask(tracer_2d, w_interp_2d, ql_2d, cb, ct, z_half)
#cleanup
del ql_2d
del w_2d
del tracer_2d
del qt
del th

print "cloud base = ", z_half[cb], " [m]"
print "cloud top  = ", z_half[ct], " [m]"
upd_frac_prof    = 1 - mask_dict["updraft"].sum(axis=0) / mask_dict["updraft"].shape[0]
env_frac_prof    = 1 - mask_dict["env"].sum(axis=0) / mask_dict["env"].shape[0]
tracer_frac_prof = 1 - mask_dict["tracer"].sum(axis=0) / mask_dict["tracer"].shape[0]
w_frac_prof      = 1 - mask_dict["w"].sum(axis=0) / mask_dict["w"].shape[0]
ql_frac_prof     = 1 - mask_dict["ql"].sum(axis=0) / mask_dict["ql"].shape[0]

plt.figure()
plt.plot(upd_frac_prof, z_half, '-', c="black", label='updraft')
plt.plot(env_frac_prof, z_half, '-', c="red",  label='environment')
plt.plot(tracer_frac_prof, z_half, '-', c="green", label='tracer')
plt.plot(w_frac_prof, z_half, '-', c="orange",  label='w')
plt.plot(ql_frac_prof, z_half, '-', c="blue", label='ql')
plt.axhline(y=z_half[cb], color='lightblue', linestyle='-', label = 'ql layer')
plt.axhline(y=z_half[ct], color='lightblue', linestyle='-')
plt.legend(loc='upper right')
plt.title(sim_name)
plt.grid(True)
plt.ylabel('height, m')
plt.xlabel('area fractions')
plt.savefig(sim_name + "_fractions.pdf")

keys_list = ["th_env", "th_upd", "qt_env", "qt_upd"]
# create dictionary for storing the updraft and environment masks for theta and qt
var_dict = {}
var_dict["th_env"] = np.ma.array(th_2d, mask=mask_dict["env"])
var_dict["th_upd"] = np.ma.array(th_2d, mask=mask_dict["updraft"])
var_dict["qt_env"] = np.ma.array(qt_2d, mask=mask_dict["env"])
var_dict["qt_upd"] = np.ma.array(qt_2d, mask=mask_dict["updraft"])

# create dictionaries for storing updraft and environment percentiles and mean profiles
mean_dict = {}
perc_dict = {}
for key in keys_list:
    mean_dict[key] = np.zeros(height.shape[0])
    perc_dict[key] = {}
    for perc in ["5", "45", "55", "95"]:
        perc_dict[key][perc] = np.zeros(height.shape[0])

for key in keys_list:
    for idx in xrange(height.shape[0]):
        if var_dict[key][:,idx].compressed().size:
            mean_dict[key][idx] = var_dict[key][:,idx].compressed().mean()
            for perc in [5, 45, 55, 95]:
                perc_dict[key][str(perc)][idx] = np.percentile(var_dict[key][:, idx].compressed(), perc)
        else:
            mean_dict[key][idx] = np.NaN
            for perc in [5, 45, 55, 95]:
                perc_dict[key][str(perc)][idx] = np.NaN

# plot qt percentiles
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
    for iz in focus_levels:
        plots[plot_it].axhline(z_half[iz], c="peachpuff", lw=2)

plots[0].plot(mean_dict["qt_env"] * 1e3,       height,       ".-", c='blue',      label='env')
plots[0].plot(mean_dict["qt_upd"][1:-2] * 1e3, height[1:-2], ".-", c='lightblue', label='upd')

plots[1].fill_betweenx(height,\
                       (perc_dict["qt_env"][ "5"]/mean_dict["qt_env"] * 100 - 100), (perc_dict["qt_env"]["95"]/mean_dict["qt_env"] * 100 - 100),\
                       color="lightblue")
plots[1].fill_betweenx(height,\
                       (perc_dict["qt_env"]["45"]/mean_dict["qt_env"] * 100 - 100), (perc_dict["qt_env"]["55"]/mean_dict["qt_env"] * 100 - 100),\
                       color="blue")

plots[2].plot(mean_dict["qt_upd"][1:-2] * 1e3 , height[1:-2], c='blue',      label='upd')
plots[2].plot(mean_dict["qt_env"] * 1e3 ,       height,       c='lightblue', label='env')

plots[3].fill_betweenx(height,\
                       (perc_dict["qt_upd"][ "5"]/mean_dict["qt_upd"] * 100 - 100), (perc_dict["qt_upd"]["95"]/mean_dict["qt_upd"] * 100 - 100),\
                       color="lightblue")
plots[3].fill_betweenx(height,\
                       (perc_dict["qt_upd"]["45"]/mean_dict["qt_upd"] * 100 - 100), (perc_dict["qt_upd"]["55"]/mean_dict["qt_upd"] * 100 - 100),\
                       color="blue")

for it in [0,2]:
    plots[it].legend()
    plots[it].set_xlabel('QT [g/kg]')
    plots[it].set_xlim(qt_range)
for it in [1,3]:
    plots[it].set_xlim(qt_spread)

plots[1].set_xlabel('QT spread around env mean [%]')
plots[3].set_xlabel('QT spread around upd mean [%]')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(sim_name)
plt.savefig(sim_name + "_full_LES_spread_qt.pdf")
plt.clf()


#plot th percentiles
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
    for iz in focus_levels:
        plots[plot_it].axhline(z_half[iz], c="peachpuff", lw=2)

plots[0].plot(mean_dict["th_env"],       height,       ".-", c='blue',      label='env')
plots[0].plot(mean_dict["th_upd"][1:-2], height[1:-2], ".-", c='lightblue', label='upd')

plots[1].fill_betweenx(height,\
                       (perc_dict["th_env"][ "5"]/mean_dict["th_env"] * 100 - 100), (perc_dict["th_env"]["95"]/mean_dict["th_env"] * 100 - 100),\
                       color="lightblue")
plots[1].fill_betweenx(height,\
                       (perc_dict["th_env"]["45"]/mean_dict["th_env"] * 100 - 100), (perc_dict["th_env"]["55"]/mean_dict["th_env"] * 100 - 100),\
                       color="blue")

plots[2].plot(mean_dict["th_upd"][1:-2], height[1:-2], c='blue',      label='upd')
plots[2].plot(mean_dict["th_env"],       height,       c='lightblue', label='env')

plots[3].fill_betweenx(height,\
                       (perc_dict["th_upd"][ "5"]/mean_dict["th_upd"] * 100 - 100), (perc_dict["th_upd"]["95"]/mean_dict["th_upd"] * 100 - 100),\
                       color="lightblue")
plots[3].fill_betweenx(height,\
                       (perc_dict["th_upd"]["45"]/mean_dict["th_upd"] * 100 - 100), (perc_dict["th_upd"]["55"]/mean_dict["th_upd"] * 100 - 100),\
                       color="blue")
for it in [0,2]:
    plots[it].legend()
    plots[it].set_xlabel('TH [K]')
    plots[it].set_xlim(th_range)

for it in [1,3]:
    plots[it].set_xlim(th_spread)

plots[1].set_xlabel('TH spread around env mean [%]')
plots[3].set_xlabel('TH spread around upd mean [%]')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(sim_name)
plt.savefig(sim_name + "_full_LES_spread_th.pdf")
plt.clf()

# dummy variables for calculating the saturation line
dummy_qt = np.linspace(qt_range[0],  qt_range[1], 1e3)
dummy_th = np.linspace(th_range[0],  th_range[1], 1e3)

# 2D distributions
fig = plt.figure(1)
fig.set_figheight(10)
fig.set_figwidth(12)
mpl.rcParams.update({'font.size': 12})
mpl.rc('lines', linewidth=3, markersize=6)

plots = []
for plot_it in range(6):
    plots.append(plt.subplot(3,2,plot_it+1))
                           #(rows, columns, number)
    plots[plot_it].set_title("z = " + str(z_half[focus_levels[plot_it]]) + " [m]")
    plots[plot_it].set_xlabel("QT [g/kg]")
    plots[plot_it].set_ylabel("TH [K]")

    iz = focus_levels[plot_it]

    hst = plots[plot_it].hist2d(\
              var_dict["qt_env"][:, iz].compressed() * 1e3,\
              var_dict["th_env"][:, iz].compressed(),\
              bins=n_bins,\
              range = [env_qt_range, env_th_range],\
              cmap=cm.Blues,\
              norm=LogNorm())
    plt.colorbar(hst[3], ax=plots[plot_it])

    dummy_T  = dummy_th * wrp.exner_c(p0_half[focus_levels[plot_it]])
    dummy_qt_star = np.zeros_like(dummy_T)
    for el_it in range(dummy_T.shape[0]):
        dummy_pv_star = wrp.pv_star(dummy_T[el_it])
        dummy_qt_star[el_it] = wrp.qv_star_c(p0_half[focus_levels[plot_it]], dummy_qt[el_it] * 1e-3, dummy_pv_star) * 1e3

    plots[plot_it].plot(dummy_qt_star, dummy_th, '-', lw=1, c='red')

    plots[plot_it].axhline(mean_dict["th_env"][iz],       c="lightgray", lw=1)
    plots[plot_it].axvline(mean_dict["qt_env"][iz] * 1e3, c="lightgray", lw=1)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(sim_name + " - 2D environment histograms")
plt.savefig( sim_name + "_full_LES_2d_hist_env.pdf")
plt.clf()

# 2D distributions
fig = plt.figure(1)
fig.set_figheight(10)
fig.set_figwidth(12)
mpl.rcParams.update({'font.size': 12})
mpl.rc('lines', linewidth=3, markersize=6)

plots = []
for plot_it in range(6):
    plots.append(plt.subplot(3,2,plot_it+1))
                           #(rows, columns, number)
    plots[plot_it].set_title("z = " + str(z_half[focus_levels[plot_it]]) + " [m]")
    plots[plot_it].set_xlabel("QT [g/kg]")
    plots[plot_it].set_ylabel("TH [K]")

    iz = focus_levels[plot_it]

    hst = plots[plot_it].hist2d(\
              var_dict["qt_upd"][:, iz].compressed() * 1e3,\
              var_dict["th_upd"][:, iz].compressed(),\
              bins=n_bins,\
              range = [upd_qt_range, upd_th_range],\
              cmap=cm.Blues,\
              norm=LogNorm())
    plt.colorbar(hst[3], ax=plots[plot_it])

    dummy_T  = dummy_th * wrp.exner_c(p0_half[focus_levels[plot_it]])
    dummy_qt_star = np.zeros_like(dummy_T)
    for el_it in range(dummy_T.shape[0]):
        dummy_pv_star = wrp.pv_star(dummy_T[el_it])
        dummy_qt_star[el_it] = wrp.qv_star_c(p0_half[focus_levels[plot_it]], dummy_qt[el_it] * 1e-3, dummy_pv_star) * 1e3

    plots[plot_it].plot(dummy_qt_star, dummy_th, '-', lw=1, c='red')

    plots[plot_it].axhline(mean_dict["th_upd"][iz],       c="lightgray", lw=1)
    plots[plot_it].axvline(mean_dict["qt_upd"][iz] * 1e3, c="lightgray", lw=1)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(sim_name + " - 2D updraft histograms")
plt.savefig( sim_name + "_full_LES_2d_hist_upd.pdf")
plt.clf()

# 2D surface plots
fig = plt.figure(1)
fig.set_figheight(10)
fig.set_figwidth(12)
mpl.rcParams.update({'font.size': 12})
mpl.rc('lines', linewidth=3, markersize=6)
plots = []
for plot_it in range(6):
    plots.append(plt.subplot(3,2,plot_it+1))
                           #(rows, columns, number)
    plots[plot_it].set_title("z = " + str(z_half[focus_levels[plot_it]]) + " [m]")

    iz = focus_levels[plot_it]

    tmp = var_dict["qt_upd"][:, iz].reshape(nx, ny)
    tmp_plot = plots[plot_it].imshow(tmp * 1e3, vmin=upd_qt_range[0], vmax=upd_qt_range[1])
    plt.colorbar(tmp_plot, ax=plots[plot_it])

plt.savefig( sim_name + "_qt_upd_2D_surfaces.pdf")
plt.clf()


fig = plt.figure(1)
fig.set_figheight(10)
fig.set_figwidth(12)
mpl.rcParams.update({'font.size': 12})
mpl.rc('lines', linewidth=3, markersize=6)
plots = []
for plot_it in range(6):
    plots.append(plt.subplot(3,2,plot_it+1))
                           #(rows, columns, number)
    plots[plot_it].set_title("z = " + str(z_half[focus_levels[plot_it]]) + " [m]")

    iz = focus_levels[plot_it]

    tmp = var_dict["qt_env"][:, iz].reshape(nx, ny)
    tmp_plot = plots[plot_it].imshow(tmp * 1e3, vmin=env_qt_range[0], vmax=env_qt_range[1])
    plt.colorbar(tmp_plot, ax=plots[plot_it])

plt.savefig( sim_name + "_qt_env_2D_surfaces.pdf")
plt.clf()


fig = plt.figure(1)
fig.set_figheight(10)
fig.set_figwidth(12)
mpl.rcParams.update({'font.size': 12})
mpl.rc('lines', linewidth=3, markersize=6)
plots = []
for plot_it in range(6):
    plots.append(plt.subplot(3,2,plot_it+1))
                           #(rows, columns, number)
    plots[plot_it].set_title("z = " + str(z_half[focus_levels[plot_it]]) + " [m]")

    iz = focus_levels[plot_it]

    tmp = var_dict["th_upd"][:, iz].reshape(nx, ny)
    tmp_plot = plots[plot_it].imshow(tmp, vmin=upd_th_range[0], vmax=upd_th_range[1])
    plt.colorbar(tmp_plot, ax=plots[plot_it])

plt.savefig( sim_name + "_th_upd_2D_surfaces.pdf")
plt.clf()


fig = plt.figure(1)
fig.set_figheight(10)
fig.set_figwidth(12)
mpl.rcParams.update({'font.size': 12})
mpl.rc('lines', linewidth=3, markersize=6)
plots = []
for plot_it in range(6):
    plots.append(plt.subplot(3,2,plot_it+1))
                           #(rows, columns, number)
    plots[plot_it].set_title("z = " + str(z_half[focus_levels[plot_it]]) + " [m]")

    iz = focus_levels[plot_it]

    tmp = var_dict["th_env"][:, iz].reshape(nx, ny)
    tmp_plot = plots[plot_it].imshow(tmp, vmin=env_th_range[0], vmax=env_th_range[1])
    plt.colorbar(tmp_plot, ax=plots[plot_it])

plt.savefig( sim_name + "_th_env_2D_surfaces.pdf")
plt.clf()
