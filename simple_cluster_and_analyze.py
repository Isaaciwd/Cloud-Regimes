# Description
# This document will open user fed data, preform clustering with either wasserstein distance or euclidean distance, and create maps of the resultant CRs. It is also possible to
# feed in premade CRs to map and skip clustering. The user can select any lat/lon box, time range, and can select to only use data from over land or only over water.
#%%
from Functions import plot_hists, plot_rfo, open_and_process
import logging as lgr

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

# If you wish to only fit data into a set of premade CRs, set this equal to a path to a numpy ndarray of premade cloud regimes of
# shape=(k, n_tau_bins * n_pressure_bins). This will skip clustering and preform analysis with the premade regimes
# If used, the below kmeans properties are ignored, and k is set to premade_cloud_regimes.shape[0]
# Using this is different from setting init in kmeans properties to a set of CRs, as that will continue clustering using the premade 
# cloud regimes and update them. This will not update the cloud regimes and skips clustering entirely. It will only fit the data into these CRs
premade_cloud_regimes = '/project/amp02/idavis/isccp_clustering/modis_and_misr/modis_testing/MODIS_emd-means_n_init5_centers_1.npy'

# kmeans properties
k=6   # number of cluster to create
tol = 30    # maximum change in inertia values between kmeans iterations to declare convergence. should be higher if using wasserstein distance
max_iter = 2   # maximum number of k-means iterations to preform for each initiation
init='k-means++'    # initialization technique for kmeans, can be 'k-means++', 'random', or initial clusters to use of shape (k, n_tau_bins * n_pressure_bins)
n_init = 2    # number of initiations of the k-means algorithm. The final result will be the initiation with the lowest calculated inertia

# Choose whether to use a euclidean or wasserstein distance
wasserstein_or_euclidean = "euclidean"

# Minimum and Maximum longitudes and latitudes entered as list, or None for entire range
lat_range = [-90,90]
lon_range = [-180,180]

# Time Range min and max as list, or None for all times present in the files
time_range = ["2003-03-01", "2004-07-01"] 

# Use data only over land or over ocean
# Set to 'L' for land only, 'O' for ocean only, or False for both land and ocean
only_ocean_or_land = 'O'
# Does this dataset have a built in variable for land fraction? if so enter as a string, otherwise cartopy will be used to mask out land or water
land_frac_var_name = None

# Logging level, set to "INFO" for information about what the code is doing, otherwise keep at "WARNING"
logging_level = 'INFO'


# Setting up logger
lgr.root.setLevel(logging_level)
# Opening data, and clustering
mat, cluster_labels, cluster_labels_temp, valid_indicies, ds = open_and_process(data_path, k, tol, max_iter, init, n_init, var_name, tau_var_name, ht_var_name, lat_var_name, lon_var_name, height_or_pressure, wasserstein_or_euclidean, premade_cloud_regimes, lat_range, lon_range, time_range, only_ocean_or_land, land_frac_var_name)
# Plotting histograms
plot_hists(cluster_labels, k, ds, ht_var_name, tau_var_name, valid_indicies, mat, cluster_labels_temp, height_or_pressure)
# Plotting RFO
plot_rfo(cluster_labels, k ,ds)
# %%

