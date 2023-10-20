"""
This file contains all functions used to assert tests to snowslide
"""
# Importations 

import numpy as np
import rasterio 
from math import *
from snowslide.snowslide_main import snowslide_base

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

def test_mass_conservation() :
    
    # We create the ideal concave dem with a flat area
    dem = ideal_concave_dem(50,2,3)

    # We store it as a .tif file 
    transform = rasterio.transform.from_origin(0, 0, 1, 1)
    crs = rasterio.crs.CRS.from_epsg(4326)
    dem_path = "ideal_concave_dem.tif"
    with rasterio.open(dem_path, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1, dtype=dem.dtype, crs=crs, transform=transform) as dst:
        dst.write(dem, 1)

    # We launch snowslide
    # Snowslide simulation
    param_routing={"routing":'mfd',"preprocessing":True}
    param_prcpt={"init":False,"SND0":None,"quantity":1,"time":0,"isotherme":2500,"zone":False,"x":None,"y":None}
    SND,convergence,SND_tot = snowslide_base(dem_path,resolution=30,param_routing=param_routing,param_prcpt=param_prcpt)
    
    total = np.sum(np.sum(SND_tot,axis=1),axis=1)
    grad = np.gradient(total)
    length = len(grad)
    test = np.zeros(length)
    if grad == test :
        return True
    else :
        return False

def catchment_area_inside_dem():
    return True




