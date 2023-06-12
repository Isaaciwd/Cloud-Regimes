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
import xarray as xr
import glob
from Functions import emd_means, euclidean_kmeans, plot_hists_k_testing, histogram_cor, create_land_mask
import logging as lgr
#%%
# Path to data to cluster
data_path = "/project/amp02/idavis/isccp_clustering/modis_and_misr/MODIS/*.nc" 

# Variable name of data to cluster in data_path
# Name of tau dimension for var_name
# Name of height/pressure dimension for var_name
var_name =  'MODIS_CLD_HISTO' 
tau_var_name =  'COT' 
ht_var_name =  'PRES' 

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
wasserstein_or_euclidean = "wasserstein"

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

# Logging level, set to "INFO" for more infromation from wassertein clustering, otherwise keep at "WARNING"
logging_level = 'WARNING'

# Setting up logger
lgr.basicConfig(level=lgr.DEBUG)
# Getting files
files = glob.glob(data_path)
# Opening an initial dataset
init_ds = xr.open_mfdataset(files[0])
# Creating a list of all the variables in the dataset
remove = list(init_ds.keys())
# Deleting the variables we want to keep in our dataset, all remaining variables will be dropped upon opening the files, this allows for faster opening of large files
remove.remove(var_name)
# If land_frac_var_name is a string, take it out of the variables to be droppe dupon opening files. If it has been entered incorrectly inform the user and proceed with TODO
if land_frac_var_name != None:
    try: remove.remove(land_frac_var_name)
    except: 
        land_frac_var_name = None
        print(f'{land_frac_var_name} variable does not exist, make sure land_frac_var_name is set correctly. Using TODO for land mask')


# opening data
ds = xr.open_mfdataset(files, drop_variables = remove)

# turning into a dataarray
ds = ds[var_name]

# Adjusting lon to run from -180 to 180 if it doesnt already
if np.max(ds.lon) > 180: 
    ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
    ds = ds.sortby(ds.lon)

# TODO MODIS has bogus values for pressure and tau in the coordinate variables, true values are listed as an attribute. This slice of code is still compatible with modis, but may not be compatible with another dataset that stores bogus coordinate values
# Reordering y dimension from lowest to graound to highest if it needs to be reordered
if height_or_pressure == 'h': 
    if ds[ht_var_name][0] > ds[ht_var_name][-1]: 
        ds.reindex({ht_var_name:ds[ht_var_name][::-1]})
if height_or_pressure == 'p':
    if ds[ht_var_name][0] < ds[ht_var_name][-1]: 
        ds.reindex({ht_var_name:ds[ht_var_name][::-1]})

# Selecting only points over ocean or points over land if only_ocean_or_land has been used
if only_ocean_or_land != False:

    # Mask out land or water with LANDFRAC variable if we have it
    if land_frac_var_name != None:
        if only_ocean_or_land == 'L': ds = ds.where(ds[land_frac_var_name] == 1)
        elif only_ocean_or_land == 'O': ds = ds.where(ds[land_frac_var_name] == 0)
        else: raise Exception('Invalid option for only_ocean_or_land: Please enter "O" for ocean only, "L" for land only, or set to False for both land and water')

    # Otherwise use cartopy
    else:
        # Creating land mask
        oh_land = create_land_mask(ds)

        # inserting new axis to make oh_land a broadcastable shape with ds
        dims = ds.dims
        for n in range(len(dims)):
            if dims[n] != 'lat' and dims[n] != 'lon':
                oh_land = np.expand_dims(oh_land, n)

        # Masking out the land or water
        if only_ocean_or_land == 'L': ds = ds.where(oh_land == 1)
        elif only_ocean_or_land == 'O': ds = ds.where(oh_land == 0)
        else: raise Exception('Invalid option for only_ocean_or_land: Please enter "O" for ocean only, "L" for land only, or set to False for both land and water')
    
# Selecting lat range
if lat_range != None:
    if ds.lat[0] > ds.lat[-1]:
        ds = ds.sel(lat=slice(lat_range[1],lat_range[0]))
    else:
        ds = ds.sel(lat=slice(lat_range[0],lat_range[1]))

# Selecting Lon range
if lon_range != None:
    if ds.lon[0] > ds.lon[-1]:
        ds = ds.sel(lon=slice(lon_range[1],lon_range[0]))
    else:
        ds = ds.sel(lon=slice(lon_range[0],lon_range[1]))

# Selecting time range
if time_range != None:
    ds = ds.sel(time=slice(time_range[0],time_range[1]))

# Selecting only valid tau and height/pressure range
# Many data products have a -1 bin for failed retreivals, we do not wish to include this
tau_selection = {tau_var_name:slice(0,9999999999999)}
# Making sure this works for pressure which is ordered largest to smallest and altitude which is ordered smallest to largest
if ds[ht_var_name][0] > ds[ht_var_name][-1]: ht_selection = {ht_var_name:slice(9999999999999,0)}
else: ht_selection = {ht_var_name:slice(0,9999999999999)}
ds = ds.sel(tau_selection)
ds = ds.sel(ht_selection)

# Selcting only the relevant data and stacking it to shape n_histograms, n_tau * n_pc
dims = list(ds.dims)
dims.remove(tau_var_name)
dims.remove(ht_var_name)
histograms = ds.stack(spacetime=(dims), tau_ht=(tau_var_name, ht_var_name))
weights = np.cos(np.deg2rad(histograms.lat.values)) # weights array to use with emd-kmeans

# Turning into a numpy array for clustering
mat = histograms.values

# Removing all histograms with 1 or more nans in them
indicies = np.arange(len(mat))
is_valid = ~np.isnan(mat.mean(axis=1))
is_valid = is_valid.astype(np.int32)
valid_indicies = indicies[is_valid==1]
mat=mat[valid_indicies]
weights=weights[valid_indicies]

if np.min(mat < 0):
    raise Exception (f'Found negative value in ds.{var_name}, if this is a fill value for missing data, convert to nans and try again')

# Setting up array to hold reults of every clustering trial
cl_trials = np.empty((n_trials,k,mat.shape[1]))

# Preform clustering with specified distance metric, n_trials times
for trial in range(n_trials):
    if wasserstein_or_euclidean == "wasserstein":
        cl_trials[trial], cluster_labels_temp, throw_away, throw_away2 = emd_means(mat, k, tol, init, n_init, ds, tau_var_name, ht_var_name, hard_stop = 45, weights = None)
    elif wasserstein_or_euclidean == "euclidean":
        cl_trials[trial], cluster_labels_temp = euclidean_kmeans(k, init, n_init, mat, max_iter)
    else: raise Exception ('Invalid option for wasserstein_or_euclidean. Please enter "wasserstein", "euclidean"')


#%%
# reshaping the cluster centers to cluster again
cl_mat = cl_trials.reshape(-1,mat.shape[1])
if wasserstein_or_euclidean == "wasserstein":
    cl, cluster_labels, throw_away, throw_away2 = emd_means(cl_mat, k, tol, init, n_init, ds, tau_var_name, ht_var_name, hard_stop = 45, weights = None)
elif wasserstein_or_euclidean == "euclidean":
    cl, cluster_labels = euclidean_kmeans(k, init='k-means++', n_init=100, mat=cl_mat, max_iter=45)
#%%
for i in range(k):
    histogram_cor(cl_mat[np.where(cluster_labels==i)])
    plot_hists_k_testing(cl_mat[np.where(cluster_labels==i)], k, ds, tau_var_name, ht_var_name, height_or_pressure)

# %%




# %%
