"""Useful functions used in the diferents versions of snowslide"""

import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import pysheds
import mayavi
from pysheds.grid import Grid

#from snowslide.snowslide import cfg

# Main functions used in the heart of snowslide simulations

def dem_flow(HNSO,grid,routing,preprocessing) :
    """ This function can preprocess the dem (or not) and compute the flow direction based on the 
    total elevation surface (dem + snow depth)

    Parameters
    ----------
    HNSO: numpy matrix
        the total elevation surface matrix (dem + SND)
    grid: ####
        the pysheds module to compute flow directions
    routing: str
        the routing method used in pysheds ('d8' or 'mfd')
    preprocessing: bolean
        activate or deactivate preprocessing
    
    Returns 
    ----------
    the flow direction matrix (numpy matrix but diferent between d8 and mfd)
    """
    
        ### DEM preprocessing: activated or not ###
    if preprocessing == True : 
        # Fill pits
        pit_filled_dem = grid.fill_pits(HNSO)
        # Fill depressions
        flooded_dem = grid.fill_depressions(pit_filled_dem)
        # Resolve flats
        inflated_dem = grid.resolve_flats(flooded_dem)

        ### Compute flow direction ###
        flow_dir = grid.flowdir(inflated_dem,routing=routing)
    else : 
        flow_dir = grid.flowdir(HNSO,routing=routing)

    return flow_dir 

def precipitations_base(path_dem,quantity,isotherme,x,y) :
    """This function initialize an ideal SND matrix based on solid precipitation information

    Parameters 
    ----------
    path_dem: str
        path to the .tif file containing the dem
    quantity: float
        amount of snow added on every point of the grid (in meters)
    isotherme: float
        altitude under which solid precipitation is considered 0 %
    x: list ([xmin,xmax])
        longitude window outside which solid precipitation is considerd 0 % (in pixels indices)
    y: list ([xmin,xmax])
        latitude window outside which solid precipitation is considerd 0 % (in pixels indices)

    Returns
    -------
    The initialization of the snow depth matrix (np matrix)
    """

    # Opening of the dem with pysheds ###
    grid = Grid.from_raster(path_dem)
    dem = grid.read_raster(path_dem) 
    
    # Initializing SND with uniform solid precipitations ###
    if x == None : 
        SND = np.full(np.shape(dem),float(quantity))
    else : 
        SND = np.full(np.shape(dem),0)
        SND[y[0]:y[1],x[0]:x[1]]=float(quantity)
    
    # Modifying SND based on the 'isotherme' chosen
    SND[np.where(dem<isotherme)[0],np.where(dem<isotherme)[1]]=0

    return SND

def slope(dem,resolution_x,resolution_y) :
    """ This function calculate a slope matrix based on the dem matrix

    Parameters
    ----------
    dem: np matrix
        topography matrix containing the altitude data
    resolution_x: float
        longitude resolution of each pixel (in meters)
    resolution_y: float
        latitude resolution of each pixel (in meters)
    
    Returns
    -------
    The matrix of the slope (np matrix)
    """

    # Initialization of the matrices in x and y directions
    n = np.shape(dem)
    n_x = np.shape(dem)[1]
    n_y = np.shape(dem)[0]
    p = np.zeros(n)
    q = np.zeros(n)

    # Remplir plutot que de recréer à chaque pas de temps (soit les donner en entrées à la fonction, soit global parameter)

    # Compute gradient components(inside)
    p[1:-1,1:-1] = ((dem[:-2,2:] + 2*dem[1:-1,2:] + dem[2:,2:]) - (dem[:-2,:-2] + 2*dem[1:-1,:-2] + dem[2:,:-2]))/(8*resolution_x)
    q[1:-1,1:-1] = ((dem[:-2,2:] + 2*dem[:-2,1:-1] + dem[:-2,:-2]) - (dem[2:,2:] + 2*dem[2:,1:-1] + dem[2:,:-2]))/(8*resolution_y)

    # Ajouter le keyword (false par défault du bord)
    
        # Compute gradient components (edges)
    n_x,n_y = n_x-1,n_y-1
    p[1:-1,0] = ((dem[:-2,1] + 2*dem[1:-1,1] + dem[2:,1]) - (dem[:-2,0] + 2*dem[1:-1,0] + dem[2:,0]))/(4*resolution_x)
    p[1:-1,n_x] = ((dem[:-2,n_x] + 2*dem[1:-1,n_x] + dem[2:,n_x]) - (dem[:-2,n_x-1] + 2*dem[1:-1,n_x-1] + dem[2:,n_x-1]))/(4*resolution_x)
    p[0,1:-1] = (dem[0,2:]- dem[0,:-2])/(2*resolution_x)
    p[n_y,1:-1] = (dem[n_y,2:]- dem[n_y,:-2])/(2*resolution_x)
    
    q[0,1:-1] = ((dem[1,:-2] + 2*dem[1,1:-1] + dem[1,2:]) - (dem[0,:-2] + 2*dem[0,1:-1] + dem[0,2:]))/(4*resolution_y)
    q[n_y,1:-1] = ((dem[n_y,:-2] + 2*dem[n_y,1:-1] + dem[n_y,2:]) - (dem[n_y-1,:-2] + 2*dem[n_y-1,1:-1] + dem[n_y-1,2:]))/(4*resolution_y)
    q[1:-1,0] = (dem[2:,0]-dem[:-2,0])/(2*resolution_y)
    q[1:-1,n_x] = (dem[2:,n_x]-dem[:-2,n_x])/(2*resolution_y)

        # Compute gradient components (corners)
    p[0,0] = (dem[0,1]-dem[0,0])/resolution_x
    q[0,0] = (dem[1,0]-dem[0,0])/resolution_y
    p[0,n_x] = (dem[0,n_x]-dem[0,n_x-1])/resolution_x
    q[0,n_x] = (dem[1,n_x]-dem[0,n_x])/resolution_y

    p[n_y,0] = (dem[n_y,1]-dem[n_y,0])/resolution_x
    q[n_y,0] = (dem[n_y,0]-dem[n_y-1,0])/resolution_y
    p[n_y,n_x] = (dem[n_y,n_x]-dem[n_y,n_x-1])/resolution_x
    p[n_y,n_x] = (dem[n_y,n_x]-dem[n_y-1,n_x])/resolution_y

    # Compute gradient from components
    gradient  = np.sqrt(p**2+q**2)

    # Compute slope from gradient
    slope = np.arctan(gradient)*(180.0 / np.pi)

    return slope

def SND_max_exponential(slope,a,c,min) : # Conseil min = 0.05
    """ Function that compute the maximal height of snow each pixel can store based on the slope. 
    The function is an exponential and parameters are estimated from 'Bernhardt & Schulz 2007'.

    Parameters
    ----------
    slope: np matrix
        Matrix that attributes a slope value to each pixel 
    a: float
        Parameter of the exponential function. Default is -0.14. 
    c: float
        Parameter of the exponential function. Default is 145
    min: float 
        Minimum snow height each pixel can store regardless of the slope. Default is 0.05.
    
    Returns
    -------
    A matrix associating to each pixel the maximum heiht of snow it can store without routing (np matrix in meters)
    """
    
    SND_max = c*np.exp(-a*slope)

    if min!=0.05 : 
        SND_max[np.where(SND_max<min)[0],np.where(SND_max<min)[1]]=min
    else : 
        SND_max[np.where(SND_max<0.05)[0],np.where(SND_max<0.05)[1]]=0.05 # Default is 0.05
        
    return SND_max

def snow_routing(SND, SND_max,flow_dir,routing) :
    """ That function routes the snow based on the routing method chosen ('mfd' or 'd8'). 
    It is called at each iteration. 

    Parameters
    ----------
    SND: np matrix
        Matrix of the snow heights associated to each pixel at iteration k
    SND_max: np matrix
        matrix associating to each pixel the maximum heiht of snow it can store based on an exponential function of slope
    flow_dir: np matrix or list of np matrices (depending on the routing method)
        Matrix of the direction and coefficient of snow that is routed to the next pixels
    routing: str
        Routing method chosen by the users ('mfd' or 'd8')
    """
    # Bibliothèques
    import numpy as np
    
    # Compute the quantity and fraction of snow that should be routed
    SNR = SND - SND_max # SNR is the quantity of snow that can be routed for each pixel
    SNR[np.where(SNR<0)[0],np.where(SNR<0)[1]]=0 # Negative values are computed when no amount of snow can be routed
    # Then every negative value is set to 0, if threshold is not reached, the snow isn't routed.  
    
    if routing=='mfd' : 
        direction_indices = {'North' : 0,'Northeast':1,'East':2,'Southeast':3,'South':4,'Southwest':5,'West':6,'Northwest':7}
        # Neighbour to the North
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir[0][2:,1:-1]*SNR[2:,1:-1] - flow_dir[0][1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the NorthEast
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir[1][2:,:-2]*SNR[2:,:-2] - flow_dir[1][1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the East
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir[2][1:-1,:-2]*SNR[1:-1,:-2] - flow_dir[2][1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the SouthEast
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir[3][:-2,:-2]*SNR[:-2,:-2] - flow_dir[3][1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the South
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir[4][:-2,1:-1]*SNR[:-2,1:-1] - flow_dir[4][1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the SouthWest
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir[5][:-2,2:]*SNR[:-2,2:] - flow_dir[5][1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the West
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir[6][1:-1,2:]*SNR[1:-1,2:] - flow_dir[6][1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the NorthWest
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir[7][2:,2:]*SNR[2:,2:] - flow_dir[7][1:-1,1:-1]*SNR[1:-1,1:-1]
    
    if routing=='d8' :
        direction_indices = {'east': 1, 'northeast': 128, 'north': 64,'northwest': 32, 'west': 16, 'southwest': 8,'south': 4, 'southeast':2}
        # Neighbour to the East
        flow_dir_E = np.copy(flow_dir)
        flow_dir_E[np.where(flow_dir!=1)[0],np.where(flow_dir!=1)[1]]=0
        flow_dir_E[np.where(flow_dir==1)[0],np.where(flow_dir==1)[1]]=1
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir_E[1:-1,:-2]*SNR[1:-1,:-2] - flow_dir_E[1:-1,1:-1]*SNR[1:-1,1:-1] 
        # Neighbour to the NorthEast
        flow_dir_NE = np.copy(flow_dir)
        flow_dir_NE[np.where(flow_dir!=128)[0],np.where(flow_dir!=128)[1]]=0
        flow_dir_NE[np.where(flow_dir==128)[0],np.where(flow_dir==128)[1]]=1
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir_NE[2:,:-2]*SNR[2:,:-2] - flow_dir_NE[1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the North
        flow_dir_N = np.copy(flow_dir)
        flow_dir_N[np.where(flow_dir!=64)[0],np.where(flow_dir!=64)[1]]=0
        flow_dir_N[np.where(flow_dir==64)[0],np.where(flow_dir==64)[1]]=1
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir_N[2:,1:-1]*SNR[2:,1:-1] - flow_dir_N[1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the NorthWest
        flow_dir_NO = np.copy(flow_dir)
        flow_dir_NO[np.where(flow_dir!=32)[0],np.where(flow_dir!=32)[1]]=0
        flow_dir_NO[np.where(flow_dir==32)[0],np.where(flow_dir==32)[1]]=1
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir_NO[2:,2:]*SNR[2:,2:] - flow_dir_NO[1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the West
        flow_dir_O = np.copy(flow_dir)
        flow_dir_O[np.where(flow_dir!=16)[0],np.where(flow_dir!=16)[1]]=0
        flow_dir_O[np.where(flow_dir==16)[0],np.where(flow_dir==16)[1]]=1
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir_O[1:-1,2:]*SNR[1:-1,2:] - flow_dir_O[1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the SouthWest
        flow_dir_SO = np.copy(flow_dir)
        flow_dir_SO[np.where(flow_dir!=8)[0],np.where(flow_dir!=8)[1]]=0
        flow_dir_SO[np.where(flow_dir==8)[0],np.where(flow_dir==8)[1]]=1
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir_SO[:-2,2:]*SNR[:-2,2:] - flow_dir_SO[1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the South
        flow_dir_S = np.copy(flow_dir)
        flow_dir_S[np.where(flow_dir!=4)[0],np.where(flow_dir!=4)[1]]=0
        flow_dir_S[np.where(flow_dir==4)[0],np.where(flow_dir==4)[1]]=1
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir_S[:-2,1:-1]*SNR[:-2,1:-1] - flow_dir_S[1:-1,1:-1]*SNR[1:-1,1:-1]
        # Neighbour to the SouthEast
        flow_dir_SE = np.copy(flow_dir)
        flow_dir_SE[np.where(flow_dir!=2)[0],np.where(flow_dir!=2)[1]]=0
        flow_dir_SE[np.where(flow_dir==2)[0],np.where(flow_dir==2)[1]]=1
        SND[1:-1,1:-1] = SND[1:-1,1:-1] + flow_dir_SE[:-2,:-2]*SNR[:-2,:-2] - flow_dir_SE[1:-1,1:-1]*SNR[1:-1,1:-1]

    return SND 

def precipitations(gdir,dem,date,density):
    """ This function initialize the SND matrix based with solid preciptation based on climate data. It is designed to 
    work properly with the glaciological model OGGM. 
    
    Parameters
    ----------
    gdir: OGGM module 
        specific to OGGM
    dem: np matrix
        Topography information previously opened with pysheds.
    date: str 
        Information about the month the user wants to model : format is 'DD-MM-YYYY'. 
    density: float
        density of the snow. Default is 400 kg/m3.

    Returns
    -------
    A initialization of the SND matrix with snow heights.
    """

    climate_data_path = gdir.get_filepath('climate_historical') # FOR OGGM : 
    #climate_data_path = "/Users/llemcf/Desktop/Stage_IGE_2023/Snowslide x OGGM/climate_historical.nc"
    with xr.open_dataset(climate_data_path) as ds_clim:
        ds_clim = ds_clim.load()
    
    # Get climate data
    temp = float(ds_clim.temp.sel(time=date))
    prcp = float(ds_clim.prcp.sel(time=date))
    ref_hgt = float(ds_clim.attrs['ref_hgt'])

    # Compute temperature field based on linear variation with altitude hypothesis 
    temp_grad = 6.5e-3 # K.m-1
    temperatures = temp - temp_grad*(dem - ref_hgt)
    
    # Compute solid precipitation field based on linear variation with altitude hypothesis (between 0°C and 2°C)
    precipitations = np.full(np.shape(dem),prcp)
    precipitations[np.where(temperatures >= 2)] = 0
    precipitations[np.where((temperatures > 0) & (temperatures < 2))] = (1 - (0.5*temperatures[np.where((temperatures > 0) & (temperatures < 2))])) * prcp # coef * prcp
    precipitations = precipitations / density

    return precipitations,temperatures

# Other functions useful to use snowslide easily

def reframe_tif(input_path,output_path=None,useas_app=False,extent=[[None,None],[None,None]],plotting=True):
    """ To quickly crop a DEM, retaining only the area of interest.
    It displays the initial DEM, and the user can then enter the x and y windows of the zone to be retained 
    so that the function can store a new DEM entitled: 'reframed_dem'. 

    Parameters
    ----------
    input_path: str
        Path to the input DEM
    output_path: str
        Path where the DEM need to be stored (complete path with the file name and format). Default is working directory.
    useas_app: Bolean
        If True, the program interacts with the user which directly enters the data it needs to perform the reframing. 
        If False, the user will need to enter the data in the function arguments.
    extent: list [[xmin,xmax],[ymin,ymax]]
        Values in pixel indices of the extent of the new DEM. Values can be chosen using the first plot displayed by the function.
    plotting: Bolean
        If True the function displays the new cropped DEM for verification. 
    
    Returns
    -------
    Displays the original DEM and performs the cropping, store the new data and displays it if chosen by the user. 
    """

    with rasterio.open(input_path) as src:
        dem_init=src.read(1)
    
        if useas_app==True:
            plt.figure()
            plt.imshow(dem_init)
            plt.colorbar()
            plt.grid()
            plt.show()
            input("Start reframing ?")
            plt.close()

            xmin = int(input("Enter the xmin value (pixel indices) :"))
            xmax = int(input("Enter the xmax value (pixel indices) :"))
            ymin = int(input("Enter the ymin value (pixel indices) :"))
            ymax = int(input("Enter the ymax value (pixel indices) :"))
            extent = [[xmin,xmax],[ymin,ymax]]

        meta = src.meta.copy()
        width = extent[0][1]-extent[0][0]
        height = extent[1][1]-extent[1][0]
        window = Window(extent[0][0], extent[1][0], width, height)
        data = src.read(window=window)
        meta.update({
            'width': width,
            'height': height,
            'transform': src.window_transform(window)
        })
    
    if output_path==None:
        folder_path = os.getcwd()
        output_path = folder_path + '/reframed_dem.tif' 

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data)
    
    if plotting == True:
        plot = input('Press y if the reframed dem needs to be plotted')

    if plot=='y' :
         
        dem_reframed = rasterio.open(output_path)
        dem_reframed = dem_reframed.read(1).astype('float64')

        plt.figure(figsize=(15,6))
        plt.imshow(dem_reframed,cmap='terrain')
        plt.colorbar(label='Altitude (m)')
        plt.grid(zorder=0)
        plt.title('reframed DEM', size=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.show()

def resampling_dem(src_path,dst_path,factor) :
    """ This function allows to resample a dem to another resolution
    It can be used to increase the speed of calculations if high resolution is not required.

    Parameters
    ----------
    src_path: str
        path where to find the source dem
    dst_path: str
        path where to store the resampled dem
    factor: float
        Factor by which to multiply the resolution. factor > 1 increases resolution and factor < 1 decreases it.

    Returns
    -------
    A new dem with a chosen new resolution.  
    """
    
    with rasterio.open(src_path) as dataset:

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * factor),
                int(dataset.width * factor)
            ),
            resampling=Resampling.bilinear
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        profile = dataset.profile
        profile.update({
        'width': dataset.width*factor,
        'height': dataset.height*factor,
        'transform': dataset.transform * dataset.transform.scale(factor),
        'dtype': 'float32'
        })

        new_dem = data[0]
        with rasterio.open(dst_path, 'w',**profile) as dst:
            dst.write(new_dem, 1)

    return new_dem

