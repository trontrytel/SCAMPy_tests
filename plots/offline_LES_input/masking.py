import numpy as np
#import pylab as plt
import pickle
import netCDF4 as nc
import scipy.stats as stats



############################-------MASKING/STATS FUNCTIONS----------#####################################

def get_tracer_mask(sc, mval):
    dims = np.shape(sc)
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]

    sc_2d = sc.reshape(nx * ny, nz)

    sc_mean_profile = np.mean(sc_2d, axis=0)
    sc_std_profile = np.add(np.std(sc_2d, axis=0), 1e-20)

    # standardize the data
    sc_2d = sc_2d - sc_mean_profile
    sc_2d = sc_2d / sc_std_profile
    # Find the mask corresponding to the scalar anomaly condition
    updraft_masked_sc = np.ma.masked_less(sc_2d, mval)
    sc_mask = updraft_masked_sc.mask
    return sc_mask, sc_2d

def get_buoyancy_mask(b, threshold):
    dims = np.shape(b)
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]

    b_2d = b.reshape(nx * ny, nz)

    b_mean_profile = np.mean(b_2d, axis=0)

    # anomalize the data
    b_2d = b_2d - b_mean_profile

    # Find the mask corresponding to the scalar anomaly condition
    updraft_masked_b = np.ma.masked_less(b_2d, threshold)
    b_mask = updraft_masked_b.mask
    return b_mask, b_2d

def get_w_mask(w, threshold):
    # read in the vertical velocity
    dims = np.shape(w)
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]
    w_2d_raw = w.reshape(nx*ny,nz)
    # Here we shift the vertical velocity to the thermodynamic points
    w_2d = np.zeros((nx*ny,nz))
    w_2d[:,0] = np.multiply(w_2d_raw[:,0],0.5)
    for k in range(1,nz):
        w_2d[:,k] =np.multiply(w_2d_raw[:,k-1] + w_2d_raw[:,k],0.5)
    # Get the mask for positive vleocity
    updraft_masked_w = np.ma.masked_less_equal(w_2d, threshold)
    w_mask = updraft_masked_w.mask
    return w_mask, w_2d

def get_ql_mask(ql,threshold, include_dcbl, z_half):

    # get the mask associated with ql >0
    dims = np.shape(ql)
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]

    ql_2d = ql.reshape(nx*ny,nz)
    mean_ql = np.mean(ql_2d, axis=0)
    ql_ind = np.argwhere(mean_ql)

    if len(ql_ind) > 0 and include_dcbl:
        zb = z_half[np.min(ql_ind)]
        zt = z_half[np.max(ql_ind)]
        h = zb + (zt-zb) * 0.25
        # print('zb, zt, h', zb, zt, h)
        for k in range(nz):
            if z_half[k]<h:
                ql_2d[:,k] = np.add(ql_2d[:,k], 1.0)


    updraft_masked_ql = np.ma.masked_less_equal(ql_2d, threshold)
    ql_mask = updraft_masked_ql.mask
    del ql_2d
    return ql_mask


def conditional_moments(var, center_flag, the_mask):
    dims = np.shape(var)
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]

    var = var.reshape(nx*ny,nz)
    if center_flag:
        mean_prof = np.mean(var,axis=0)
        var = var - mean_prof
    masked_var = np.ma.array(var,mask=the_mask)
    # get the mean
    mask_var_mean = np.ma.mean(masked_var,axis=0)
    mask_var_var = np.ma.var(masked_var,axis=0)
    del masked_var

    return mask_var_mean, mask_var_var, masked_var


def conditional_moments2(var, center_flag, the_mask):
    dims = np.shape(var)
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]

    var = var.reshape(nx*ny,nz)

    if center_flag:
        mean_prof = np.mean(var,axis=0)
        var = var - mean_prof
    var2 = np.multiply(var,var)

    masked_var = np.ma.array(var, mask=the_mask)
    masked_var2 = np.ma.array(var2, mask=the_mask)
    # get the mean
    mask_var_mean = np.ma.mean(masked_var,axis=0)
    mask_var2_mean = np.ma.mean(masked_var2,axis=0)
    del var2

    return mask_var_mean, mask_var2_mean, masked_var

def conditional_product(var_a,var_b):
    product = np.multiply(var_a,var_b)
    return np.ma.mean(product,axis=0)



def conditional_correlation(var1,var2):
    dims = np.shape(var1)
    nz = dims[1]

    correlation  = np.empty(nz)
    for k in range(nz):
        corr_matrix = np.ma.corrcoef(var1[:,k],var2[:,k])
        correlation[k] = corr_matrix[0,1]

    #mask the correlation
    correlation = np.ma.masked_invalid(correlation)

    return correlation


