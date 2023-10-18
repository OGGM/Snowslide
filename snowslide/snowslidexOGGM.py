from utils import *
from oggm import tasks, utils, workflow, graphics

""" 
SnowslidexOGGM is designed to couple snowslide algorithm with OGGM workflow and allows to add an avalanche module 
to the Mass Balance model of OGGM. 
"""

def snowslidexOGGM(gdir,epsilon=0.001,date=None,density=400,
                     param_expo={"a":0.14,"c":145,"min":0.05},
                     param_routing={"routing":'mfd',"preprocessing":True}) :
    """ This function operates the gravitationnal transport of the snow from an initial map of snow heights and a dem.

    Parameters
    ----------
    gdir ::py:class:`oggm.GlacierDirectory`
        the glacier directory to process containing all the data
    epsilon: float
        Condition to get out the loop (convergence is considered reached when indicateur < epsilon). Default is 0.001.
    date: str
        format : 'YYYY-MM-DD'. Default is the '2018-01-01'. 
    param_expo: dictionary
        The maximum snow height each pixel can store is computed as an exponential function of the slope. Parameters are the following:
            SND_max=c*exp(a*slope)
        a: float
            Default is -0.14. 
        c: float
            Default is 145
        min: float
            Default is 0.05.
    param_routing: dictionary
        Parameters used by pysheds to realize the routing of the snow. 
        routing: str 
            routing method ('mfd' or 'd8')
        preprocessing: bolean
            activate or deactivate preprocessing of the DEM. Deactivate it affects convergence. 

    Returns
    -------
    Final matrix of the snow height (np matrix in meters). It is stored in 'gridded data' xarray dataset under 'avalanches' name.
    """

                ### Initialization ###
    
    # Variable definition needed in Snowslide
    path_dem = gdir.dir + '/dem.tif'
    gridded_data_path = gdir.get_filepath('gridded_data')
    resolution = gdir.grid.dx
    if date == None:
        date = '2018-01-01'

    iter = 0
    indicateur = float('inf')

        # Importation of dem using pysheds 
    grid = Grid.from_raster(path_dem)
    dem = grid.read_raster(path_dem)

        # initialization of snow quantities
    SND = precipitations(gdir,dem,date,density)

    print('Variables have been initialized, launching the simulation...')
    print(f'As a reminder the simulation has been launched with the {param_routing["routing"]} routing algorithm on {date}')

                ### Core part of the code ###                                                    

    iter = 0
    convergence=[]
    while indicateur > epsilon :    
        SND1=SND.copy() 
        HNSO = dem + SND # total elevation surface (HNSO) is recalculated at each iteration
        flow_dir = dem_flow(HNSO,grid,param_routing["routing"],param_routing["preprocessing"])
        HNSO_slope = slope(HNSO,resolution,resolution)
        SND_max = SND_max_exponential(HNSO_slope,param_expo["a"],param_expo["c"],param_expo["min"])
        SND = snow_routing(SND,SND_max,flow_dir,param_routing["routing"])
        
        iter = iter + 1

                    ### Exit conditions ###
        # 1st exit condition 
        indicateur = np.sum((SND-SND1)**2) # Exit condition when the L2 norm of the distance 
        # between the matrices of two iterations converge towards 0. 
    
        # 2nd exit condition if indicateur converge towards a constant that is not 0. 
        if iter > 5 : 
            speed = np.sum(np.gradient(np.array(convergence)[-5:]))
            if speed < 0 and abs(speed)<1e-2:
                indicateur = 0

    print("The algorithm converged in :",iter," iterations")
    
    # Saving the output as xarray DataArray in 'gridded data' dataset
    with xr.open_dataset(gridded_data_path) as ds:
        ds = ds.load()
        # Keeping only avalanches contributing to glacier MB (ie above the glacier) using glacier mask
        SND_glacier = SND.copy()
        SND_glacier[np.where(ds.glacier_mask==0)]=np.nan
        # convert m to mm w.e
        SND_glacier = SND_glacier * (density/1000)*1000
        ds['avalanches'] = (('y','x'),SND_glacier)
        ds.to_netcdf(gridded_data_path)

def add_to_flowline(gdir):
    """ Function that integrates avalanches snow heights to an OGGM flowline
    
    Parameters
    ----------
    gdir ::py:class:`oggm.GlacierDirectory`
        the glacier directory to process containing all the data

    Returns
    -------
    writes a new .csv file to disk ('elevation_band_flowline') containing the snow heights along the flowline
    and convert it to an OGGM flowline. 
    """
        
    tasks.elevation_band_flowline(gdir, bin_variables=['avalanches'])
    tasks.fixed_dx_elevation_band_flowline(gdir, bin_variables=['SND_snowslide'], preserve_totals=True)
