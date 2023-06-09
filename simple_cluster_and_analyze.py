# Data Requirements
# Must have lat and lon as coordintate names for latitude and longitude TODO maybe fix this

# Description
# this document will open user fed data, preform clustering with either wasserstein distance or euclidean distance, and create maps of the resultant CRs. It is also possible to
# feed in premade CRs to map and skip clustering, or to continue clustering to refine these CRs. The user can select any lat/lon box, time range, and can select to only 
# use data from over land or only over water.
#%%
from time import time
import numpy as np
import wasserstein
import matplotlib.pyplot as plt
from scipy import sparse
import xarray as xr
import matplotlib as mpl
from numba import njit
from sklearn.cluster import KMeans
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
from sklearn.cluster import KMeans
import glob
from math import ceil
from shapely.geometry import Point
import cartopy
from shapely.prepared import prep
from numba import njit

global num_iter, n_samples, data, ds, ht_var_name, tau_var_name, k, height_or_pressure

# Plot the CR cluster centers
def plot_hists(cl, cluster_labels):

    # # Converting fractional data to percent to plot properly
    # if np.max(cl) <= 1:
    #     cl *= 100

    # setting up plots
    ylabels = ds[ht_var_name].values
    xlabels = ds[tau_var_name].values
    X2,Y2 = np.meshgrid(np.arange(len(xlabels) + 1), np.arange(len(ylabels+1)))
    p = [0,0.2,1,2,3,4,6,8,10,15,99]
    cmap = mpl.colors.ListedColormap(['white', (0.19215686274509805, 0.25098039215686274, 0.5607843137254902), (0.23529411764705882, 0.3333333333333333, 0.6313725490196078), (0.32941176470588235, 0.5098039215686274, 0.6980392156862745), (0.39215686274509803, 0.6, 0.43137254901960786), (0.44313725490196076, 0.6588235294117647, 0.21568627450980393), (0.4980392156862745, 0.6784313725490196, 0.1843137254901961), (0.5725490196078431, 0.7137254901960784, 0.16862745098039217), (0.7529411764705882, 0.8117647058823529, 0.2), (0.9568627450980393, 0.8980392156862745,0.1607843137254902)])
    norm = mpl.colors.BoundaryNorm(p,cmap.N)
    plt.rcParams.update({'font.size': 12})
    fig_height = 1 + 10/3 * ceil(k/3)
    fig, ax = plt.subplots(figsize = (17, fig_height), ncols=3, nrows=ceil(k/3), sharex='all', sharey = True)

    aa = ax.ravel()
    boundaries = p
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    aa[1].invert_yaxis()

    # creating weights area for area weighted RFOs
    weights = cluster_labels.stack(z=('time','lat','lon')).lat.values
    weights = np.cos(np.deg2rad(weights))
    weights = weights[valid_indicies]
    indicies = np.arange(len(mat))

    # Plotting each cluster center
    for i in range (k):

        # Area Weighted relative Frequency of occurence calculation
        total_rfo_num = cluster_labels == i 
        total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat)))
        total_rfo_denom = cluster_labels >= 0
        total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat)))
        total_rfo = total_rfo_num  / total_rfo_denom * 100
        total_rfo = total_rfo.values

        # Area weighting each histogram belonging to a cluster and taking the mean
        # if clustering was preformed with wasserstein distance and area weighting on, mean of i = cl[i], however if clustering was preformed with
        # conventional kmeans or wasseerstein without weighting, these two will not be equal
        indicies_i = indicies[np.where(cluster_labels_temp == i)]
        mean = mat[indicies_i] * weights[indicies_i][:,np.newaxis]
        mean = np.sum(mean, axis=0) / np.sum(weights[indicies_i])
        mean = mean.reshape(len(xlabels),len(ylabels)).T           # reshaping into original histogram shape
        if np.max(mean) <= 1:                                      # Converting fractional data to percent to plot properly
            mean *= 100

        im = aa[i].pcolormesh(X2,Y2,mean,norm=norm,cmap=cmap)
        aa[i].set_title(f"CR {i+1}, RFO = {np.round(total_rfo,1)}%")

    # setting titles, labels, etc
    if height_or_pressure == 'p': fig.supylabel(f'Cloud-top Pressure', fontsize = 12, x = 0.09 )
    if height_or_pressure == 'h': fig.supylabel(f'Cloud-top Height', fontsize = 12, x = 0.09  )
    # fig.supxlabel('Optical Depth', fontsize = 12, y=0.26 )
    cbar_ax = fig.add_axes([0.95, 0.38, 0.045, 0.45])
    cb = fig.colorbar(im, cax=cbar_ax, ticks=p)
    cb.set_label(label='Cloud Cover (%)', size =10)
    cb.ax.tick_params(labelsize=9)
    #aa[6].set_position([0.399, 0.125, 0.228, 0.215])
    #aa[6].set_position([0.33, 0.117, 0.36, 0.16])
    #aa[-2].remove()

    bbox = aa[1].get_position()
    p1 = bbox.p1
    p0 = bbox.p0
    fig.suptitle(f'Cloud Regimes', x=0.5, y=p1[1]+(1/fig_height * 0.5), fontsize=15)
    print(p1[1])

    bbox = aa[-2].get_position()
    p1 = bbox.p1
    p0 = bbox.p0
    fig.supxlabel('Optical Depth', fontsize = 12, y=p0[1]-(1/fig_height * 0.5) )

    print(1 + 10/3 * ceil(k/3))


    # Removing extra plots
    for i in range(ceil(k/3)*3-k):
        aa[-(i+1)].remove()

# Plot RFO maps of the CRss
def plot_rfo(cluster_labels):
    
    COLOR = 'black'
    mpl.rcParams['text.color'] = COLOR
    mpl.rcParams['axes.labelcolor'] = COLOR
    mpl.rcParams['xtick.color'] = COLOR
    mpl.rcParams['ytick.color'] = COLOR
    plt.rcParams.update({'font.size': 10})
    fig_height = 2.2 * ceil(k/2)
    fig, ax = plt.subplots(ncols=2, nrows=int(k/2 + k%2), subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (10,fig_height))#, sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.13, hspace=0.05)
    aa = ax.ravel()
    plt.rcParams['figure.dpi'] = 500

    X, Y = np. meshgrid(ds.lon,ds.lat)

    # Plotting the rfo of each cluster
    tot_rfo_sum = 0 
    
    for cluster in range(k): #range(0,k+1):
        # Calculating rfo
        rfo = np.sum(cluster_labels==cluster, axis=0) / np.sum(cluster_labels >= 0, axis=0) * 100
        # tca_explained = np.sum(cluster_labels == cluster) * np.sum(init_clusters[cluster]) / total_cloud_amnt * 100
        # tca_explained = round(float(tca_explained.values),1)
        aa[cluster].set_extent([-180, 180, -90, 90])
        aa[cluster].coastlines()
        mesh = aa[cluster].pcolormesh(X, Y, rfo, transform=ccrs.PlateCarree(), rasterized = True, cmap="GnBu",vmin=0,vmax=100)
        #total_rfo = np.sum(cluster_labels==cluster) / np.sum(cluster_labels >= 0) * 100
        # total_rfo_num = np.sum(cluster_labels == cluster * np.cos(np.deg2rad(cluster_labels.lat)))
        total_rfo_num = cluster_labels == cluster 
        total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat)))
        total_rfo_denom = cluster_labels >= 0
        total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat)))

        total_rfo = total_rfo_num  / total_rfo_denom * 100
        tot_rfo_sum += total_rfo
        aa[cluster].set_title(f"CR {cluster+1}, RFO = {round(float(total_rfo),1)}", pad=4)
        # aa[cluster].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        # x_label_plot_list = [4,5,6]
        # y_label_plot_list = [0,2,4,6]
        # if cluster in x_label_plot_list:


        if cluster % 2 == 0:
            aa[cluster].set_yticks([-60,-30,0,30,60], crs=ccrs.PlateCarree())        
            lat_formatter = LatitudeFormatter()
            aa[cluster].yaxis.set_major_formatter(lat_formatter)

    #aa[7].set_title(f"Weathersdfasdfa State {i+1}, RFO = {round(float(total_rfo),1)}", pad=-40)
    cb = plt.colorbar(mesh, ax = ax, anchor =(-0.28,0.83), shrink = 0.6)
    cb.set_label(label = 'RFO (%)', labelpad=-3)

    x_ticks_indicies = np.array([-1,-2])

    if k%2 == 1:
        aa[-1].remove()
        x_ticks_indicies -= 1

        #aa[-2].set_position([0.27, 0.11, 0.31, 0.15])

    # plotting x labels on final two plots
    aa[x_ticks_indicies[0]].set_xticks([-120,-60,0,60,120,], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    aa[x_ticks_indicies[0]].xaxis.set_major_formatter(lon_formatter)
    aa[x_ticks_indicies[1]].set_xticks([-120,-60,0,60,120,], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    aa[x_ticks_indicies[1]].xaxis.set_major_formatter(lon_formatter)

    bbox = aa[1].get_position()
    p1 = bbox.p1
    plt.suptitle(f"CR Relative Frequency of Occurence", x= 0.43, y= p1[1]+(1/fig_height * 0.5))#, {round(cl[cluster,23],4)}")

    plt.show()

# K-means algorithm that uses wasserstein distance
def emd_means(mat, k, tol, init, n_init, hard_stop = 45, weights = None):

    # A function to convert mat into the form reqired by the wasserstein package
    @njit()
    def stacking(position_matrix, centroids):
        centroid_list = []

        for i in range(len(centroids)):
            x = np.empty((3,len(mat[0]))).T
            x[:,0] = centroids[i]
            x[:,1] = position_matrix[0]
            x[:,2] = position_matrix[1]
            centroid_list.append(x)

        return centroid_list

    n, d = mat.shape

    # checking for a weights array to prefrom a weighted kmeans with
    if type(weights) == np.ndarray: 
        weighted = True
        mat_w = mat * weights[:,None]
        print("AREA WEIGHTED")
    else: weighted = False

    centroid_labels = np.arange(k, dtype=np.int32)  # (k, )
  
    centroid_tracking = []
    inertia_tracking = np.zeros(n_init)

    # Setting the number of tau and height dimensions for each data set
    n1 = len(ds[tau_var_name])
    n2 = len(ds[ht_var_name])

    # Calculating the max distance between two points to be used as hyperparameter in EMD
    # This is not necesarily the only value for this variable that can be used, see Wasserstein documentation
    # on R hyper-parameter for more information
    R = (n1**2+n2**2)**0.5

    # Creating a flattened position matrix to pass wasersstein.PairwiseEMD
    position_matrix = np.zeros((2,n1,n2))
    position_matrix[0] = np.tile(np.arange(n2),(n1,1))
    position_matrix[1] = np.tile(np.arange(n1),(n2,1)).T
    position_matrix = position_matrix.reshape(2,-1)

    # Initialising wasserstein.PairwiseEMD
    emds = wasserstein.PairwiseEMD(R = R, norm=True, dtype=np.float32, verbose=0, num_threads=162)

    # Rearranging mat to be in the format necesary for wasserstein.PairwiseEMD
    events = stacking(position_matrix, mat)

    # Preforming n_init initiations of the kmeans algorithm, and then keeping the best initiation as a result
    for init_number in range(n_init):
        emd_inertia_list = []

        # Using Kmeans++ if init  == True
        if init == 'k-means++':
            init_clusters = np.zeros((k, len(mat[0])))
            init_clusters[0] = mat[np.random.randint(0,len(mat))]
            centroid_list = stacking(position_matrix, init_clusters)

            dists_ar = np.full((len(mat),k-1), np.inf)
            t = 0
            for i in range(k-1):
                x = time()
                emds(events,centroid_list[i:i+1])
                t += time() - x
                dists_ar[:,i] = emds.emds().squeeze()
                weights_kpp = np.min(dists_ar, axis=1)
                weights_kpp = weights_kpp / np.sum(weights_kpp)

                choice = np.random.choice(np.arange(len(mat)), 1, p=weights_kpp)
                init_clusters[i+1] = mat[choice]
                centroid_list = stacking(position_matrix, init_clusters)

            print(f"{round(t,1)} Seconds for k-means++ initialization")

            centroids = init_clusters
        
        # Using array entered as init as initial centroids if init is an ndarray
        elif type(init) == np.ndarray:
            if init.shape != (k,d): raise Exception ('init array must be shape (k, n_tau_bins * n_pressure_bins)')
            centroids = init
            n_init = 1 # Doing more than one init in this case is useless, as they will all have the same result
        
        # Otherwise using random initiation
        elif init == 'random':
            # Randomly picking k observations to use as initial clusters
            centroids = mat[np.random.choice(n, k, replace=False)]  # (k, d)

        else:
            raise Exception (f'Enter valid option for init. Enter "k-means++" to use kmeans++, "random" for random initiation, or set equal to a (k, n_tau_bins * n_pressure_bins) shaped ndarray to use as initial clusters. You entered {init}')

        iter = 0

        # initializing inertia_diff so loop will run, true values is calculated after the second iteration
        inertia_diff = tol+1 
        t = 0
        while inertia_diff >= tol:

            # ASSIGNMENT STEP
            centroid_list = stacking(position_matrix, centroids)
            emds(events, centroid_list)
            distances = emds.emds()
            labels = np.argmin(distances, axis=1)

            #calculating emd_inertia
            onehot_matrix = labels[:,None] == centroid_labels  # (n, k)
            if weighted: emd_inertia = np.sum((distances[onehot_matrix]*weights/np.sum(weights))**2)
            else: emd_inertia = np.sum(distances[onehot_matrix]**2)
            emd_inertia_list.append(emd_inertia)
            
            s = time()
            # Updating cluster centroids
            if weighted == False:
                b_data, b_oh = np.broadcast_arrays(  # (n, k, d), (n, k, d)
                    mat[:, None], onehot_matrix[:, :, None])
                centroids =  b_data.mean(axis=0, where=b_oh)  # (k, d)
            
            # Updating cluster centers as area weighted average
            if weighted == True:
                centroids = np.zeros((k,d))
                for i in range (k):
                    centroids[i] = np.sum(mat_w[np.where(labels == i)], axis = 0) / np.sum(weights[np.where(labels == i)])

            t += time() - s

            # Calculate change in inertia from last step
            if iter > 0:
                inertia_diff = emd_inertia_list[-2] - emd_inertia_list[-1]
                print(inertia_diff)

            iter += 1
        
            # Check if we've reached the hard stop on number of iterations
            if iter == hard_stop:
                print(F"WARNING: HARD STOP = {hard_stop} REACHED")
                print(f"tol = {tol}, final inertia = {emd_inertia}")
                break

        print(f"{iter} iterations until convergence with tol = {tol} ")
        print(f"Weighted = {weighted}: time spent updating centroids per iter = {round(t/iter,2)}")

        centroid_tracking.append(centroids)
        inertia_tracking[init_number] = emd_inertia

    # retreiving the cluster centers that had the lowest inertia
    best_result = np.argmin(inertia_tracking)

    # recaluclating cluster labels to the final updated cluster centers
    centroid_list = stacking(position_matrix, centroids)
    emds(events, centroid_list)
    distances = emds.emds()
    labels = np.argmin(distances, axis=1)

    return centroid_tracking[best_result], labels, inertia_tracking, centroid_tracking

# Conventional kmeans using sklearn
def euclidean_kmeans(k, init, n_init, mat, max_iter):
    # Seting up kmeans nd fitting the data
    kmeans = KMeans(n_clusters=k, init = init, n_init = n_init, max_iter=max_iter).fit(mat)
    # Retreiving cluster labels
    cluster_labels_temp = kmeans.labels_
    # Retreiving cluster centers
    cl = kmeans.cluster_centers_

    return cl, cluster_labels_temp

# Compute cluster labels from precomputed cluster centers with appropriate distance
def precomputed_clusters(mat, cl, wasserstein_or_euclidean):

    if wasserstein_or_euclidean == 'euclidean':
        cluster_dists = np.sum((mat[:,:,None] - cl.T[None,:,:])**2, axis = 1)
        cluster_labels_temp = np.argmin(cluster_dists, axis = 1)

    if wasserstein_or_euclidean == 'wasserstein':

        # A function to convert mat into the form required for the EMD calculation
        @njit()
        def stacking(position_matrix, centroids):
            centroid_list = []

            for i in range(len(centroids)):
                x = np.empty((3,len(mat[0]))).T
                x[:,0] = centroids[i]
                x[:,1] = position_matrix[0]
                x[:,2] = position_matrix[1]
                centroid_list.append(x)

            return centroid_list
        
        # setting shape
        n1 = len(ds[tau_var_name])
        n2 = len(ds[ht_var_name])

        # Calculating the max distance between two points to be used as hyperparameter in EMD
        # This is not necesarily the only value for this variable that can be used, see Wasserstein documentation
        # on R hyper-parameter for more information
        R = (n1**2+n2**2)**0.5

        # Creating a flattened position matrix to pass wasersstein.PairwiseEMD
        position_matrix = np.zeros((2,n1,n2))
        position_matrix[0] = np.tile(np.arange(n2),(n1,1))
        position_matrix[1] = np.tile(np.arange(n1),(n2,1)).T
        position_matrix = position_matrix.reshape(2,-1)

        # Initialising wasserstein.PairwiseEMD
        emds = wasserstein.PairwiseEMD(R = R, norm=True, dtype=np.float32, verbose=1, num_threads=162)

        # Rearranging mat to be in the format necesary for wasserstein.PairwiseEMD
        events = stacking(position_matrix, mat)
        centroid_list = stacking(position_matrix, cl)
        emds(events, centroid_list)
        distances = emds.emds()
        labels = np.argmin(distances, axis=1)

        cluster_labels_temp = np.argmin(distances, axis=1)
        
    return cluster_labels_temp

# Create a one hot matrix where lat lon coordinates are over land using cartopy
def create_land_mask(ds):
    
    land_110m = cartopy.feature.NaturalEarthFeature('physical', 'land', '110m')
    land_polygons = list(land_110m.geometries())
    land_polygons = [prep(land_polygon) for land_polygon in land_polygons]

    lats = ds.lat.values
    lons = ds.lon.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = [Point(point) for point in zip(lon_grid.ravel(), lat_grid.ravel())]

    land = []
    for land_polygon in land_polygons:
        land.extend([tuple(point.coords)[0] for point in filter(land_polygon.covers, points)])

    landar = np.asarray(land)
    lat_lon = np.empty((len(lats)*len(lons),2))
    oh_land = np.zeros((len(lats)*len(lons)))
    lat_lon[:,0] = lon_grid.flatten()
    lat_lon[:,1] = lat_grid.flatten()

    @njit()
    def test (oh_land, lat_lon, landar):
        for i in range(len(oh_land)):
            check = lat_lon[i] == landar
            if np.max(np.sum(check,axis=1)) == 2:
                oh_land[i] = 1
        return oh_land
    oh_land = test (oh_land, lat_lon, landar)


    oh_land=oh_land.reshape((len(lats),len(lons)))

    return oh_land


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
k=6   # number of cluster to create
tol = 30    # maximum changne in inertia values between kmeans iterations to declare convergence. should be higher if using wasserstein distance
max_iter = 30   # maximum number of k-means iterations to preform for each initiation
init='k-means++'    # initialization technique for kmeans, can be 'k-means++', 'random', or initial clusters to use of shape (k, n_tau_bins * n_pressure_bins)
n_init = 1    # number of initiations of the k-means algorithm. The final result will be the initiation with the lowest calculated inertia

# Choose whether to use a euclidean or wasserstein distance kmeans algorithm
wasserstein_or_euclidean = "euclidean"

# Set this equal to a numpy ndarray of premade cloud regimes (shape=(k, n_tau_bins * n_pressure_bins)) to skip clustering and preform analysis with the premade regimes
# If used, the above kmeans properties are ignored, and k is set to premade_cloud_regimes.shape[0]
# Using this is different from setting init in kmeans properties, as that will continue clustering using the premade cloud regimes and update them. This will not update the cloud regimes
# and skips clustering entirely. It will only fit the data into these CRs
premade_cloud_regimes = None

# Minimum and Maximum longitudes and latitudes entered as list, or None for entire range
lat_range = [-90,90]
lon_range = [-180,180]

# Time Range min and max, or None for all time
time_range = ["2003-03-01", "2004-07-01"] 

# Use data only over land or over ocean
# Set to 'L' for land only, 'O' for ocean only, or False for both land and ocean
only_ocean_or_land = 'O'
# Does this dataset have a built in variable for land fraction? if so enter as a string, otherwise TODO will be used to mask out land or water
land_frac_var_name = None

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
    # lon_grid, lat_grid = np.meshgrid(ds.lon,ds.lat)
    # water = globe.is_land(lat_grid, lon_grid) # creating an array that's 1 over land, and 0 over water
    # if np.max(ds.lon) > 180: water = np.roll(water, 180, axis=1) # shifting back
    

    oh_land = create_land_mask(ds)

    # inserting new axis to make water a broadcastable shape with ds
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

# Use premade clusters to calculate cluster labels (using specified distance metric) if they have been provided
if type(premade_cloud_regimes) == np.ndarray:
    cl = premade_cloud_regimes
    k = len(premade_cloud_regimes)
    if premade_cloud_regimes.shape != (k,len(ds[tau_var_name]) * len(ds[ht_var_name])):
        raise Exception (f'premade_cloud_regimes is the wrong shape. premade_cloud_regimes.shape = {premade_cloud_regimes.shape}, but must be shpae {(k,len(ds.tau_var_name) * len(ds.ht_var_name))} to fit the loaded data')
    print('Using premade cloud regimes:')
    cluster_labels_temp = precomputed_clusters(mat, cl, wasserstein_or_euclidean)

# Otherwise preform clustering with specified distance metric
else:
    if wasserstein_or_euclidean == "wasserstein":
        cl, cluster_labels_temp, il, cl_list = emd_means(mat, k=k, tol=tol, init=init, n_init = n_init, hard_stop=max_iter, weights=weights)
    elif wasserstein_or_euclidean == "euclidean":
        cl, cluster_labels_temp = euclidean_kmeans(k, init, n_init, mat, max_iter)
    else: raise Exception ('Invalid option for wasserstein_or_euclidean. Please enter "wasserstein", "euclidean", or a numpy ndarray to use as premade cloud regimes and preform no clustering')

# taking the flattened cluster_labels_temp array, and turning it into a datarray the shape of ds.var_name, and reinserting NaNs in place of missing data
cluster_labels = np.full(len(indicies), np.nan, dtype=np.int32)
cluster_labels[valid_indicies]=cluster_labels_temp
cluster_labels = xr.DataArray(data=cluster_labels, coords={"spacetime":histograms.spacetime},dims=("spacetime") )
cluster_labels = cluster_labels.unstack()

# Plotting histograms
plot_hists(cl, cluster_labels)
# Plotting RFO
plot_rfo(cluster_labels)
# %%

# %%
