"""
This file contains all functions used to assert tests to snowslide
"""
import os
import numpy as np
import rasterio 
from math import *
from snowslide.snowslide_main import snowslide_base
from snowslide.functions import *

def test_mass_conservation1() :
    """ This function tests mass is well conserved by the algorithm.
    """

    def ideal_concave_dem(nb_pixels,mean_slope,factor) :
        """ This function creates an ideal concave dem to test mass conservation of snowslide

        Parameters
        ----------
        nb_pixels: float
            Number of pixels of the DEM we want to create
        mean_slope: float
            Approximate slope we want to see on our DEM (depending on what needs to be tested)
        factor: float
            We need to create a flat area at the bottom of the DEM. The factor is the relation between width and radius of this area. 
            factor = width/radius
    
        Outputs
        -------
        Z: np matrix
            Ideal concave dem we wanted to create
        """

    # Defining some important values to create DEM
    # We assume each pixel has a 30m resolution
        resolution = 30
        width = nb_pixels*resolution
        diameter = width/factor
        height = mean_slope*((width/2)-diameter)

        # Create the grid 
        x = np.arange(-width/2,width/2,resolution)
        y = np.arange(-width/2,width/2,resolution)
        X, Y = np.meshgrid(x, y)
        
        # Create the altitude matrix
        max = np.max(x)**2
        coef = height/max
        Z = coef*X**2 + coef*Y**2

        # Create the flat area necessary for testing the convergence of snowslide
        index = int((width - diameter)/(2*resolution))+1
        flat_value=Z[int(nb_pixels/2),index]

        for i in range(nb_pixels) :
            for j in range(nb_pixels) :
                if ((i-(nb_pixels/2))**2 + (j-(nb_pixels/2))**2) < (diameter/(2*resolution))**2 : # Pour le cercle du milieu en gros 
                    Z[i,j] = flat_value

        return Z
    
    # We create the ideal concave dem with a flat area
    dem = ideal_concave_dem(50,2,3)

    # We store it as a .tif file 
    transform = rasterio.transform.from_origin(0, 0, 1, 1)
    crs = rasterio.crs.CRS.from_epsg(4326)
    dem_path = "data/ideal_concave_dem.tif"
    with rasterio.open(dem_path, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1, dtype=dem.dtype, crs=crs, transform=transform) as dst:
        dst.write(dem, 1)

    # We launch snowslide
    # Snowslide simulation
    param_routing={"routing":'mfd',"preprocessing":True}
    SND0 = np.full(np.shape(dem),1)
    SND = snowslide_base(dem_path,SND0=SND0,param_routing=param_routing)
    
    # Compute total masses
    initial_mass = np.sum(SND0)
    final_mass = np.sum(SND)

    #delete dem file
    if os.path.exists(dem_path):
        os.remove(dem_path)

    assert initial_mass == final_mass, f"Total snow mass is not conserved between beginning ({initial_mass}) and end of convergence ({final_mass})."

def test_mass_conservation2() :
    """ This function tests mass is well conserved by the algorithm.
    """

    def create_ideal_dem(alt_max,alt_min,res) :
        """ This function creates an ideal dem

        Parameters 
        ----------
        alt_max: float
            Maximum altitude for the dem 
        alt_min: float
            Minimum altitude for the dem
        res: float
            Resolution of the dem 
        
        Output
        ------
            dem: numpy.ndarray
                Resulting dem
        """

        nb_row = 25
        nb_col = 50
        dem = np.full((nb_row+2,nb_col+2),alt_max)
        vector1 = np.arange(alt_max,alt_min,-res)
        vector2 = np.zeros(nb_row)
        vector = np.concatenate((vector1,vector2))
        dem[1:-1,1:-1] = np.tile(vector, (nb_row, 1))

        return dem 
    
    # We create the ideal concave dem with a flat area
    dem = create_ideal_dem(750,0,30)

    # We store it as a .tif file 
    transform = rasterio.transform.from_origin(0, 0, 1, 1)
    crs = rasterio.crs.CRS.from_epsg(4326)
    dem_path = "data/ideal_dem.tif"
    with rasterio.open(dem_path, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1, dtype=dem.dtype, crs=crs, transform=transform) as dst:
        dst.write(dem, 1)

    # We launch snowslide
    # Snowslide simulation
    param_routing={"routing":'mfd',"preprocessing":True}
    SND0 = np.zeros(np.shape(dem))
    SND0[1:-1,1:-1] = 1 
    SND = snowslide_base(dem_path,SND0=SND0,param_routing=param_routing)
    
    # Compute total masses
    initial_mass = np.sum(SND0)
    final_mass = np.sum(SND)

    #delete dem file
    if os.path.exists(dem_path):
        os.remove(dem_path)

    assert initial_mass == final_mass, f"Total snow mass is not conserved between beginning ({initial_mass}) and end of convergence ({final_mass})."

def test_maxSND_respected():
    """ This function tests if the snowdepth after convergence are are actually less than the maximum retention capacity set. 
    The results are accepted to within a certain margin, as the algorithm may leave the loop before it has fully converged. 
    """
    
    dem_path = "data/DEM_Mt_blanc_Talefre.tif"
    dem = rasterio.open(dem_path).read(1)
    
    param_routing={"routing":'mfd',"preprocessing":True}
    SND0 = np.full(np.shape(dem),float(1))
    SND = snowslide_base(dem_path,SND0=SND0,param_routing=param_routing,epsilon=1e-5)
    
    resx,resy = rasterio.open(dem_path).res
    slope = slope(dem+SND,resx,resy)

    parameters = {'a':-0.14,'c':145,'min':0.05}
    SND_max = parameters['c']*np.exp(parameters['a']*slope)
    SND_max[np.where(SND_max < parameters['min'])[0],np.where(SND_max < parameters['min'])[1]] = parameters['min']

    test = np.copy(SND_max-SND)
    test[np.where(test>0)] = 0

    # assertion 
    margin = 1e-2
    minimum = np.min(test)
    if minimum < 0 :
        assert abs(minimum) < abs(margin)

def test_run_snowslide():
    """ This function tests that snowslide runs without any error using a real DEM of Talefre basin in Mont-Blanc mountain range
    """

    dem_path = "data/DEM_Mt_blanc_Talefre.tif"
    dem = rasterio.open(dem_path).read(1)
    param_routing={"routing":'mfd',"preprocessing":True}
    SND0 = np.full(np.shape(dem),float(1))
    bolean=True
    try:
        snowslide_base(dem_path,SND0=SND0,param_routing=param_routing)
    except:
        bolean=False
    assert bolean==True

def test_flat_holdSnow():
    """ This function tests if flat areas receive more snow than steep areas after convergence. 
    The distinction between both areas is 40°. 
    """

    def create_ideal_dem(alt_max,alt_min,res) :
        vector1 = np.arange(alt_max,alt_min,-res)
        vector2 = np.zeros(25)
        vector = np.concatenate((vector1,vector2))
        matrix = np.tile(vector,(25,1))

        return matrix 

    dem = create_ideal_dem(750,0,30)
    transform = rasterio.transform.from_origin(0, 0, 20, 20)
    crs = rasterio.crs.CRS.from_epsg(4326)
    dem_path = "/Users/llemcf/Desktop/Stage_IGE_2023/1ideal_dem.tif"
    with rasterio.open(dem_path, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1, dtype=dem.dtype, crs=crs, transform=transform) as dst:
            dst.write(dem, 1)

    param_routing={"routing":'mfd',"preprocessing":True}
    SND0 = np.zeros(np.shape(dem))
    SND0[1:-1,1:-1] = 1

    # Simulation using snowslide 
    SND = snowslide_base(dem_path,SND0=SND0,param_routing=param_routing)
    slope = slope(dem,20,20)

    steep_area = np.sum(SND[np.where(slope >= 40)])
    flat_area = np.sum(SND[np.where(slope < 40)])

    assert steep_area < flat_area, ("Snow has accumulated more on steep areas.")

def test_slope_limits() :
    """ This function checks that the slope calculation gives values between 0 and 90 degrees 
    """
    dem_path = "/Users/llemcf/Desktop/Stage_IGE_2023/GitHub_snowslide/snowslide/test/data/DEM_Mt_blanc_Talefre.tif"
    dem = rasterio.open(dem_path).read(1)
    res_x , res_y = rasterio.open(dem_path).res 
    slope = slope(dem,res_x,res_y)
    
    assert np.min(slope) > 0, ("Error, the slope has values under 0°.")
    assert np.max(slope) < 90, ("Error, the slope has values over 90°.")