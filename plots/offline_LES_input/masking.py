import math  as mt
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

def reshape_var(var):
    """
    Reshape 3D variable into 2D variable
    """
    dims = np.shape(var)
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]

    var_2d = var.reshape(nx * ny, nz)
    return var_2d

def calculate_cloud_base_top(ql_2d, ql_tr = 1e-8):
    """
    Calculate cloud base and cloud top index based on ql data.
    A level is considered "cloudy" if a mean ql at this level is greater than
    the threshold value (1e-8 by default).
    The lowest cloudy index is cloud base and the highest is cloud top.
    If there is no cloud, cloud base and cloud top indices are set to NaN.

    Input:  2d field of cloud water
    Output: height indices of cloud base and cloud top

    """
    ql_2d_mean = np.mean(ql_2d, axis=0)

    idx_ql = np.where(ql_2d_mean >= ql_tr)

    if idx_ql[0].size != 0:
        cb_idx = np.min(idx_ql)
        ct_idx = np.max(idx_ql)
    else:
        cb_idx = np.NaN
        ct_idx = np.NaN

    print "CB_idx = ", cb_idx
    print "CT_idx = ", ct_idx

    return (cb_idx, ct_idx)

def interpolate_w(w_2d_full):
    """
    Interpolate vertical velocity from cell edges to cell centers
    """
    w_2d_half = np.zeros_like(w_2d_full)

    w_2d_half[:, 0] = w_2d_full[:, 0] / 2.

    for idx in range(1, np.shape(w_2d_half)[1], 1):
        w_2d_half[:,idx] = 0.5 * (w_2d_full[:, idx] + w_2d_full[:, idx-1])

    return w_2d_half

def updraft_env_mask(tracer_2d, w_interp_2d, ql_2d, cb, ct, z_half, ql_tr = 1e-8):
    """
    Find updraft and environment masks.
    The algorithm is described in Couvreux et al 2010 (DOI 10.1007/s10546-009-9456-5)

    Input:  2D tracer field,
            2D vertical velocity field (interpolated to the same locations as other variables)
            2D liquid water field
            indices of cloud base and cloud top
            1D array with height
    Output: updraft and environment masks

    """
    updraft_mask = np.ones_like(tracer_2d) #mask = 1 -> False, mask = 0 True
    tracer_mask = np.ones_like(tracer_2d)
    w_mask = np.ones_like(tracer_2d)
    ql_mask = np.ones_like(tracer_2d)
    nxy = np.shape(tracer_2d)[0]
    nz = np.shape(tracer_2d)[1]

    sigma_sum = 0.
    z_ql = 0.
    cloud_flag = False
    if np.isnan(cb) == False and np.isnan(ct) == False:
        z_ql = z_half[cb] + 0.25 * (z_half[ct] - z_half[cb])
        cloud_flag = True

    print "z_ql = ", z_ql

    tracer_mean = np.mean(tracer_2d, axis=0)
    tracer_square_mean = np.mean(tracer_2d * tracer_2d, axis=0)
    tracer_variance = tracer_square_mean - tracer_mean * tracer_mean
    assert(tracer_variance.all() >= 0)
    tracer_std = np.sqrt(tracer_variance)

    for k in range(nz):
        sigma_sum += tracer_std[k]
        sigma_min = sigma_sum/(k+1.0) * 0.05 # threshold from the paper

        for i in range(nxy):
            if tracer_std[k] >= sigma_min:
                if tracer_2d[i,k] - tracer_mean[k] >= tracer_std[k]:
                    updraft_mask[i,k] = 0
                    tracer_mask[i,k] = 0
            # TODO - I think the paper condition should also include this
            #        But it's not done in Pycles
            #else:
            #    if tracer_2d[i,k] - tracer_mean[k] >= sigma_min:
            #        updraft_mask[i,k] = 0

            if w_interp_2d[i,k] <= 0.:
                updraft_mask[i,k] = 1

            if cloud_flag:
                if z_half[k] >= z_ql and z_half[k] <= z_half[ct]:
                    if ql_2d[i,k] < ql_tr:
                        updraft_mask[i,k] = 1

    env_mask = 1 - updraft_mask
    return updraft_mask, env_mask


