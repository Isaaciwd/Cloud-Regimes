# This script is designed to test the robustness of the clustering. The user enters a data path, 
# any data restrictions such as lat/lon bounding box, time range, k-means properties, and a number of trials to preform. The clustering
# is then preformed n_trials times creating k * n_trials CRs. These are then clustered with the same values of k. If the clustering parameters 
# are robust, this should return k clusters each with n_trials members, where each member is very similar within clusters. For example, if a 
# stratocumulus CR is created, that stratocumulus CR should look cery similar in each of the clustering trials. To examine this, 
# this script plots the results of the second clustering step. It shows each group of k cluster centers, and a correlation matrix between each
# of those clusters. Ideally, there should be n_trials histograms, that look very similar with high coeficients of correlation. 
#%%
# from time import time
import numpy as np
from Functions import emd_means, euclidean_kmeans, plot_hists_k_testing, histogram_cor, open_and_process
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
tol = 400    # maximum changne in inertia values between kmeans iterations to declare convergence. should be higher if using wasserstein distance
max_iter = 2   # maximum number of k-means iterations to preform for each initiation
init='k-means++'    # initialization technique for kmeans, can be 'k-means++', 'random', or initial clusters to use of shape (k, n_tau_bins * n_pressure_bins)
n_init = 2    # number of initiations of the k-means algorithm. The final result will be the initiation with the lowest calculated inertia

# k sensitivity testing properties
n_trials = 4 # how many times to preform the clustering with above properties

# Choose whether to use a euclidean or wasserstein distance kmeans algorithm
wasserstein_or_euclidean = "euclidean"

# Minimum and Maximum longitudes and latitudes entered as list, or None for entire range: Ex [-65,65]
lat_range = [-90,90]
lon_range = [-180,180]

# Time Range min and max, or None for all time, entered as list of str: Ex. ["2003-03-01", "2004-07-01"] or ['2003','2007']
time_range = ["2003-03-01", "2004-07-01"] 

# Use data only over land or over ocean
# Set to 'L' for land only, 'O' for ocean only, or False for both land and ocean
only_ocean_or_land = 'L'
# Does this dataset have a built in variable for land fraction? if so enter variable name as a string, otherwise cartopy will be used to mask out land or water
land_frac_var_name = None

# Logging level, set to "INFO" for information about what the code is doing, otherwise keep at "WARNING"
logging_level = 'INFO'

# Setting up logger
lgr.basicConfig(level=lgr.DEBUG)
# Avoid creation of large chunks with dask
dask.config.set({"array.slicing.split_large_chunks": False})
# Automatically setting premade_cloud_regimes to none because this file does not need them. Do not Change.
premade_cloud_regimes = None

# Opening and preprocessing data
mat, valid_indicies, ds, histograms = open_and_process(data_path, k, tol, max_iter, init, n_init, var_name, tau_var_name, ht_var_name, lat_var_name, lon_var_name, height_or_pressure, wasserstein_or_euclidean, premade_cloud_regimes, lat_range, lon_range, time_range, only_ocean_or_land, land_frac_var_name, cluster = False)

# Setting up array to hold reults of every clustering trial
cl_trials = np.empty((n_trials,k,mat.shape[1]))

# Preform clustering with specified distance metric, n_trials times and record the resultant cloud regimes each time
for trial in range(n_trials):
    if wasserstein_or_euclidean == "wasserstein":
        cl_trials[trial], cluster_labels_temp, throw_away, throw_away2 = emd_means(mat, k, tol, init, n_init, ds, tau_var_name, ht_var_name, hard_stop = 45, weights = None)
    elif wasserstein_or_euclidean == "euclidean":
        cl_trials[trial], cluster_labels_temp = euclidean_kmeans(k, init, n_init, mat, max_iter)
    else: raise Exception ('Invalid option for wasserstein_or_euclidean. Please enter "wasserstein", "euclidean"')


#%%
# Reshaping the cluster centers to cluster again
cl_mat = cl_trials.reshape(-1,mat.shape[1])

# Finally, clustering all of the resultant cloud regimes from the previous step
if wasserstein_or_euclidean == "wasserstein":
    cl, cluster_labels, throw_away, throw_away2 = emd_means(cl_mat, k, tol, init, n_init, ds, tau_var_name, ht_var_name, hard_stop = 45, weights = None)
elif wasserstein_or_euclidean == "euclidean":
    cl, cluster_labels = euclidean_kmeans(k, init='k-means++', n_init=100, mat=cl_mat, max_iter=45)
#%%
# Plotting
for i in range(k):
    histogram_cor(cl_mat[np.where(cluster_labels==i)])
    plot_hists_k_testing(cl_mat[np.where(cluster_labels==i)], k, ds, tau_var_name, ht_var_name, height_or_pressure)

# %%




# %%
