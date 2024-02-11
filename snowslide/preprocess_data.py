"""Functions that automatize the downloading and preprocessing of precipitations data for the user"""

import os
import cdsapi
import rasterio
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Proj, transform
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Useful functions to preprocess the dems

def get_coordinates_central_point(dem_path) :
    """This function aims to compute the lon/lat coordinates of the center of a raster dem

    Parameters
    ----------
    dem_path: str
        Path to the dem

    Outputs
    -------
    lon,lat : tuple of floats
        Geographical coordinates in ° of the center of the dem
    """
    with rasterio.open(dem_path) as src:
        cols, rows = src.width // 2, src.height // 2
        crs = src.crs
        x, y = src.xy(rows, cols)
        in_proj = Proj(crs)
        out_proj = Proj(init='epsg:4326')
        lon, lat = transform(in_proj, out_proj, x, y)

    return lon,lat

def reframe_tif(
    input_path,
    output_path=None,
    useas_app=False,
    extent=[[None, None], [None, None]],
    plotting=True,
):
    """To quickly crop a DEM, retaining only the area of interest.
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
        dem_init = src.read(1)

        if useas_app == True:
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
            extent = [[xmin, xmax], [ymin, ymax]]

        meta = src.meta.copy()
        width = extent[0][1] - extent[0][0]
        height = extent[1][1] - extent[1][0]
        window = Window(extent[0][0], extent[1][0], width, height)
        data = src.read(window=window)
        meta.update(
            {
                "width": width,
                "height": height,
                "transform": src.window_transform(window),
            }
        )

    if output_path == None:
        folder_path = os.getcwd()
        output_path = folder_path + "/reframed_dem.tif"

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data)

    if plotting == True:
        plot = input("Press y if the reframed dem needs to be plotted")

    if plot == "y":
        dem_reframed = rasterio.open(output_path)
        dem_reframed = dem_reframed.read(1).astype("float64")

        plt.figure(figsize=(15, 6))
        plt.imshow(dem_reframed, cmap="terrain")
        plt.colorbar(label="Altitude (m)")
        plt.grid(zorder=0)
        plt.title("reframed DEM", size=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.show()

def crop_tif_from_coordinates_window(input_path,minx,maxx,miny,maxy,output_path=None):
    # Opening data
    src = rasterio.open(input_path)
    # Converting coordinates in index
    row_start, col_start = src.index(minx,maxy)
    row_stop, col_stop = src.index(maxx, miny)
    # Cropping
    window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start) 
    cropped_dem = src.read(1,window=window)
    # Store dem as new tif data if required
    if output_path is not None :
        with rasterio.open(output_path, 'w', driver='GTiff', width=window.width, height=window.height, count=src.count, dtype=src.dtypes[0], crs=src.crs, transform=src.window_transform(window)) as dst:
            dst.write(cropped_dem,1)

    return cropped_dem

def resampling_dem(src_path, dst_path, factor,resampling="bilinear"):
    """This function allows to resample a dem to another resolution
    It can be used to increase the speed of calculations if high resolution is not required.

    Parameters
    ----------
    src_path: str
        path where to find the source dem
    dst_path: str
        path where to store the resampled dem
    factor: float
        Factor by which to multiply the resolution. factor > 1 increases resolution and factor < 1 decreases it.
    resampling: str
        Rasterio method to resample. Default is bilinear for continuous data. Nearest is useful for non continuous data such as masks. 

    Returns
    -------
    A new dem with a chosen new resolution.
    """

    with rasterio.open(src_path) as dataset:
        # resample data to target shape
        if resampling=='bilinear': 
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    round(dataset.height * factor),
                    round(dataset.width * factor),
                ),
                resampling=Resampling.bilinear,
            )
        if resampling=='nearest':
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    round(dataset.height * factor),
                    round(dataset.width * factor),
                ),
                resampling=Resampling.nearest,
            )
         
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
        )

        profile = dataset.profile
        profile.update(
            {
                "width": dataset.width * factor,
                "height": dataset.height * factor,
                "transform": dataset.transform * dataset.transform.scale(1 / factor),
                "dtype": "float32",
            }
        )

        new_dem = data[0]
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(new_dem, 1)

    return new_dem

def store_output_as_raster(output_path,input_path,matrix,epsg="EPSG:4326") :
    """This function allows to store a matrix as .tif file using the georeference of the dem"""
    
    with rasterio.open(input_path) as src :
        crs = src.crs
        transformation = src.transform
        if str(src.crs)!= epsg:
            transformation, width, height = calculate_default_transform(src.crs, epsg, src.width, src.height, *src.bounds)
            resampling = Resampling.nearest
            with rasterio.open(output_path, 'w', driver='GTiff', crs=epsg, transform=transformation, width=width, height=height,count=src.count, dtype=src.dtypes[0]) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transformation,
                    dst_crs=epsg,
                    resampling=resampling)
            crs = rasterio.open(output_path).crs
            transformation = rasterio.open(output_path).transform

        with rasterio.open(output_path,'w',driver='GTiff', height=matrix.shape[0], width=matrix.shape[1],
                        count=1, dtype=matrix.dtype, crs=crs, transform=transformation) as dst:
            dst.write(matrix, 1)

# Download and preprocess s2m data (SAFRAN reanalysis)
def initialize_snowslide_from_safran(dem_path, ds_paths, massif_id=3, frequency="M", snow_density=150):
    """Function that creates SND0 matrices to initialize snowslide from SAFRAN precipitation data

    Parameters
    ----------
    dem_path: Str
        Path to the dem file over which the user wants to initialize snow heights
    ds_path: list
        List of the paths to anual xarray files downloaded through aeris-data web page
    massif_id: int
        Id of the mountain range we want to use the data from
    frequency: str
        Specifies the frequency over which snowslide is used.
        'D' : Day
        'W' : Week
        'M' : Month
        'Q' : quarter
        'A' : Year
    You can also specify a diferent frequency with for example '2M' that means '2 months'.

    Outputs
    -------
    It stores in a specified folder a numpy matrix containing an SND initialization on a monthly frequency

    Improvements to be made :
    ------------------------
        - user should have the possibility to choose daily, weekly, monthly or annual frequence ...
        - Values are always averaged within a whole mountain range based on 'massif_id' --> should be possible to either choose 'nearest station data'
        - It must be verified that this work on all sort of SAFRAN dataset (add try and if loops to be less specific)
        - It could be interesting to download directly the data from Aerius data portal

    """

    # (1) - Preprocessing the SAFRAN netcdf data

    def preprocess_SAFRAN_precipitations(ds_path, massif_id):
        """Function that preprocess SAFRAN data in order to prepare initialization of precipitations matrices for Snowslide
        SAFRAN data is available at : "https://www.aeris-data.fr/en/landing-page/?uuid=865730e8-edeb-4c6b-ae58-80f95166509b#v2020.2"

        Parameters
        ----------
        ds_path: str
            Path to the xarray file downloaded through aeris-data web page
        massif_id: int
            Id of the mountain range we want to use the data from

        outputs
        -------
        ds_snowf: xarray dataarray
            Dataarray containing the Snowfall values in kg/m2 for each month and each altitude band of 300m over a hydrological year.
        """
        # Charges xarray dataset
        ds = xr.load_dataset(ds_path)
        if "massif" in ds.coords:
            ds = ds.sel(massif=massif_id)
            ds = ds.drop(["massif"])

        # Removing bad variables
        if "isoZeroAltitude" in ds:
            ds = ds.drop_vars(["isoZeroAltitude"])
        if "rainSnowLimit" in ds:
            ds = ds.drop_vars(["rainSnowLimit"])

        # using altitude as coordinate
        ds = ds.set_coords("ZS")

        # Constructing the altitude intervals (bin vector)
        n1 = int(np.min(ds.ZS))
        n2 = int(np.max(ds.ZS))
        altitude_bin = np.arange(n1, n2 + 1, 300)

        # Realizing binning operation by averaging values by altitude bands over the whole 'massif' (mountain range)
        ds = ds.groupby_bins("ZS", altitude_bin).mean("Number_of_points")

        # Summing Snowfall rates to obtain values at some chosen frequency
        ds_resampled = ds.resample(time=frequency).sum(dim="time")
        dates = pd.DatetimeIndex(ds_resampled.time)
        if "A" in frequency:  # Yearly frequency
            # jours_par_an = np.array(dates.days_in_year)
            ds_snowf = (
                ds_resampled.Snowf * 365 * 24
            )  # Méthod 'En attendant mieux' car 365 ne prend pas en compte les années bisextiles mais on commet une très faible erreur !!
        if "M" in frequency:  # Monthly frequency
            jours_par_mois = np.array(dates.days_in_month)
            ds_snowf = ds_resampled.Snowf * jours_par_mois.reshape(-1, 1) * 24
        if "W" in frequency:  # Weekly frequency
            ds_snowf = ds_resampled.Snowf * 24 * 7
        if "D" in frequency:  # Daily frequency
            ds_snowf = ds_resampled.Snowf * 24

        # We make sure we don't have repetition with dates
        ds_snowf = ds_snowf.sel(
            time=slice(
                f"{ds.attrs['time_coverage_start'][:4]}-08-01",
                f"{round(float(ds.attrs['time_coverage_start'][:4]))+1}-07-31",
            )
        )
        ds_snowf = ds_snowf.assign_coords(
            {"ZS_bins": np.arange(1, ds.ZS_bins.shape[0] + 1)}
        )

        return ds_snowf

    ds_list = []
    for i in range(len(ds_paths)):
        ds_snowf = preprocess_SAFRAN_precipitations(ds_paths[i], massif_id=massif_id)
        ds_list.append(ds_snowf)
    ds = xr.concat([elt for elt in ds_list], dim="time", coords="different")

    # (2) - Creating matrices to initialize Snowslide

    # Importing DEM data
    src = rasterio.open(dem_path)
    dem = src.read(1)

    # Initializing precipitation matrices
    precipitations = np.zeros((ds.time.shape[0], np.shape(dem)[0], np.shape(dem)[1]))

    nb_months = ds.time.shape[0]
    nb_bins = ds.ZS_bins.shape[0]
    for t in range(nb_months):
        for h in range(nb_bins):
            precipitations[t][
                np.where((dem >= h * 300) & (dem < (h + 1) * 300))
            ] = float(ds.sel(ZS_bins=h + 1).isel(time=t).values)

    # Getting SND in m instead of kg/m2
    precipitations = precipitations / snow_density

    return precipitations

# Download and preprocess ERA5 data
def download_ERA5_data(start_date,end_date,dem_path,
                       variable=['2m_temperature','geopotential','total_precipitation'],
                       name = 'download_era5_data.nc'
                       ) :
    """This function allows the downloading of Copernicus datasets trough the copernicus climate change dataset center

    Parameters
    ----------
    start_date: str
        Start date of the dataset in format 'YYYY-MM-DD'
    end_date: str
        End date of the dataset in format 'YYYY-MM-DD'
    area: list
        List of coordinates of the area from which you wish to retrieve climatic data.
        format : [Northern latitude, Western longitude, Southern latitude, Eastern longitude]
    variable: list
        List of the variables extracted from the datacenter. By default, variable = ['2m_temperature','geopotential','total_precipitation']
    name: str
        Name of the netcdf file stored locally on the user computer. By default, name = 'download_era5_data.nc'
    """
    
    # List of years to select
    years = [str(i) for i in range(int(start_date[:4]),int(end_date[:4])+1)]
    # List of month to select
    month = ['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12']
    if len(years) == 1:
        month = [str(i) for i in range(int(start_date[5:7]),int(end_date[5:7])+1)]
    # List of days
    day = ['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12','13', '14', '15',
            '16', '17', '18','19', '20', '21','22', '23', '24','25', '26', '27','28', '29', '30','31']
    if len(month) == 1:
        day = [str(i) for i in range(int(start_date[8:10]),int(end_date[8:10])+1)]
    # Get the area from dem_path
    lon,lat = get_coordinates_central_point(dem_path)
    area = [round(lat+0.2,2),round(lon-0.2,2),round(lat-0.2,2),round(lon+0.2,2)]

    c = cdsapi.Client()

    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variable,
                'year': years,
                'month': month,
                'day': day,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': area,
            },
            name)
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")  

def initilialize_snowslide_from_era5(dem_path,ds_path,frequency="M",snow_d=200,grad=-6.5e-3):
    """Function that creates SND0 matrices to initialize snowslide from ERA5 data
    
    Parameters
    ----------
    dem_path: Str
        Path to the dem file over which the user wants to initialize snow heights
    frequency: str
        Specifies the frequency over which snowslide is used.
        'D' : Day
        'W' : Week
        'M' : Month
        'Q' : Quarter
        'A' : Year
    You can also specify a diferent frequency with for example '2M' that means '2 months'.
    snow_d: float
        Snow density. Default is 200 kg/m3
    grad: float
        Vertical temperature gradient. Default is -6.5 °C/km

    Outputs
    -------
    time: list (np array)
        time coordinates of the output
    temperature: np array
        Snow heights initialization at each time step
"""
    # Importing dem
    dem = rasterio.open(dem_path).read(1)
    # Importing weather data and resampling to a chosen frequency
    ds = xr.open_dataset(ds_path)
    ds['time'] = xr.decode_cf(ds).time
    ds = ds.resample(time=frequency).mean(dim='time')

    # Choosing only variable for the nearest lon/lat point to the center of dem
    dem_lon,dem_lat = get_coordinates_central_point(dem_path)
    lon,lat = np.meshgrid(ds.longitude.values,ds.latitude.values)
    distance = np.sqrt((lon-dem_lon)**2 + (lat-dem_lat)**2)
    index = np.argmin(distance)
    lat_id,lon_id = np.unravel_index(index, distance.shape)
    ds = ds.isel(longitude=lon_id,latitude=lat_id)

    # Initializing the matrices
    temperature = []
    precipitation = []
    alt = float(ds.z.isel(time=0))/9.81 # Altitude computed from geopotential assuming g = 9.81 m.s-2
    time = ds.time.values

    # Creating init matrices at each time period
    i = 0
    for elt in time:
        t = float(ds.t2m.sel(time=elt).values) - 273.15 
        temp = (dem - alt)*grad + t
        temperature.append(temp)

        sf = ds.tp.sel(time=elt).values
        prcpt = np.zeros(np.shape(dem))
        prcpt[temperature[i] < 0] = sf
        # Going back to a snow height instead of water height using density
        prcpt = prcpt*(1000/snow_d) # Assuming water density is 1000 kg/m3
        precipitation.append(prcpt)

        i+=1

    return time, np.array(precipitation)