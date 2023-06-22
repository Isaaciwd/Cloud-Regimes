# This script is designed to test a range of number of clusters (k) to decide on a value of k. The user enters a data path, 
# any data restrictions such as lat/lon bounding box, time range, k-means properties, and a range of values of k to test. 
# The script then preforms clustering for each values of K and creates two correlation matrices. One showing the correlation between 
# each cluster center, and one showing the correlation between each CRs spatial distribution. If an increase in k has produced two CRs that
# are highly correlated with each other, than this value of k may be too high, and the lower value should be considered.= as the final value of k.
# for our paper (Davis and Medeiros) we preformed this analysis with euclidean k-means, and then used the resultant value of k with
# wasserstein k-means to derive our final CRs
#%%
import numpy as np
import xarray as xr
from Functions import emd_means, euclidean_kmeans, plot_hists, plot_rfo, histogram_cor, spacial_cor, open_and_process
import logging as lgr
import dask

#%%
# Path to data to cluster
data_path = "/project/amp02/idavis/isccp_clustering/modis_and_misr/MODIS/*.nc" 

# Variable name of data to cluster in data_path
# Name of tau dimension for var_name
# Name of height/pressure dimension for var_name
var_name =  'MODIS_CLD_HISTO' 
tau_var_name =  'COT' 
ht_var_name =  'PRES'
lat_var_name = 'lat'
lon_var_name = 'lon'

# Does this dataset use cloud top height or cloud top pressure? enter "h" for height or "p" for pressure
height_or_pressure = 'p'

# kmeans properties
k=4   # number of cluster to create
tol = 30    # maximum changne in inertia values between kmeans iterations to declare convergence. should be higher if using wasserstein distance
max_iter = 30   # maximum number of k-means iterations to preform for each initiation
init='k-means++'    # initialization technique for kmeans, can be 'k-means++', 'random', or initial clusters to use of shape (k, n_tau_bins * n_pressure_bins)
n_init = 1    # number of initiations of the k-means algorithm. The final result will be the initiation with the lowest calculated inertia


# k sensitivity testing properties
k_range = [3,8] # minimum and maximum values for k to test

# Plot the CR centers and rfo maps? or just the correlation matricies
plot_cr_centers = True
plot_rfo_graphs = False

# Choose whether to use a euclidean or wasserstein distance kmeans algorithm
wasserstein_or_euclidean = "euclidean"

# Minimum and Maximum longitudes and latitudes entered as list, or None for entire range
lat_range = [-90,90]
lon_range = [-180,180]

# Time Range min and max, or None for all time
time_range = ["2003-03-01", "2004-07-01"] 

# Use data only over land or over ocean
# Set to 'L' for land only, 'O' for ocean only, or False for both land and ocean
only_ocean_or_land = 'L'
# Does this dataset have a built in variable for land fraction? if so enter as a string, otherwise cartopy will be used to mask out land or water
land_frac_var_name = None

# Logging level, set to "INFO" for information about what the code is doing, otherwise keep at "WARNING"
logging_level = 'INFO'

# Setting up logger
lgr.root.setLevel(logging_level)
# Avoid creation of large chunks with dask
dask.config.set({"array.slicing.split_large_chunks": False})
# Automatically setting premade_cloud_regimes to none because this file does not need them. Do not Change.
premade_cloud_regimes = None

# Opening and preprocessing data
mat, valid_indicies, ds, histograms = open_and_process(data_path, k, tol, max_iter, init, n_init, var_name, tau_var_name, ht_var_name, lat_var_name, lon_var_name, height_or_pressure, wasserstein_or_euclidean, premade_cloud_regimes, lat_range, lon_range, time_range, only_ocean_or_land, land_frac_var_name, cluster = False)

# Preform clustering with specified distance metric, for all values of k in k_range
for k in np.arange(k_range[0], k_range[1]+1):
    if wasserstein_or_euclidean == "wasserstein":
        cl, cluster_labels_temp, il, cl_list = emd_means(mat, k=k, tol=tol, init=init, n_init = n_init, hard_stop=max_iter, weights=weights)
    elif wasserstein_or_euclidean == "euclidean":
        cl, cluster_labels_temp = euclidean_kmeans(k, init, n_init, mat, max_iter)
    else: raise Exception ('Invalid option for wasserstein_or_euclidean. Please enter "wasserstein", "euclidean"')

    # Reshaping cluster_labels_temp to original shape of ds and reinserting NaNs in the original places, so spatial correlations can be calcualated
    cluster_labels = np.full(len(histograms), np.nan, dtype=np.int32)
    cluster_labels[valid_indicies]=cluster_labels_temp
    cluster_labels = xr.DataArray(data=cluster_labels, coords={"spacetime":histograms.spacetime},dims=("spacetime") )
    cluster_labels = cluster_labels.unstack()

    # Plotting
    histogram_cor(cl)
    spacial_cor(cluster_labels,k)

    if plot_cr_centers:
        plot_hists(cluster_labels, k, ds, ht_var_name, tau_var_name, valid_indicies, mat, cluster_labels_temp, height_or_pressure)

    if plot_rfo_graphs:
        plot_rfo(cluster_labels)
    
# %%
