# A simple parametrisation of snow transport and deposition by avalanches for OGGM.

The **Snowslide** python package was initiated by Guillem Carcanade, intern in 
the glaciology team of IGE Grenoble based on the algorithms described by 
Bernhardt & Schulz, 2010 and Gruber, 2007. It was further updated by Marin Kneib and 
Fabien Maussion for use in OGGM.  

This package provides a simple parametrisation of snow transport and deposition 
by avalanches. It has been designed for applications in glaciological modeling, 
and specifically to be used through the OGGM workflow. While the package provides 
methods for use in conjunction with OGGM, the basic functions can also be called 
and used independently. Snowslide can then work in a wide variety of contexts that 
include gravitational snow transport. Snowslide operates the estimated redistribution 
of snow by avalanches using only two inputs : 
- a dem given as a path to a .tif file
- information about initial snow depths 

The next steps goes into more details about how Snowslide works and should be used.

## Installation 

Snowslide relies on basic Python librairies for data manipulation. Nonetheless, 
the routing of the snow is calculated by an external Python module called pysheds
(docs : https://github.com/mdbartos/pysheds/). Running snowslide therefore requires
installing pysheds v0.5 (pip install pysheds=0.5).

The list of all snowslide dependencies is given as follow : 
- pysheds v0.5
- rasterio
- numpy
- xarray
- pandas
- matplotlib
- os
- datetime

These libraries must be installed to run snowslide. Snowslide can then be installed
using pip : 

    $ pip install -e git+https://github.com/OGGM/Snowslide.git

This should clone the snowslide GitHub repository and install it as a python
package in the active virtual environment chosen. 

Snowslide works independently but is also configured to work with the most recent version of OGGM (v1.6.3).

## Package structure

#### Directory root (``./``)

- ``.gitignore``: for git users only
- ``LICENSE.txt``: (https://help.github.com/articles/licensing-a-epository/) license of the code
- ``README.md``: this page
- ``pyproject.toml``: this is what makes your package installable by ``pip``. It
  contains a set of simple instructions regarding e.g. the name of the package,
  its version number, or where to find command line scripts.
  
#### The actual package (``./snowslide``)

- ``__init__.py``: tells Python that the directory is a package and enables
  the  "dotted module names"  import syntax. It is often empty but here
  we added some entry points to the package's API and the version string.
- ``snowslide_main.py``: main module that operates the simulation
- ``functions.py``: various functions used in the main module and defined separatly 
- ``oggm_snowslide_compat_minimal.py``: module recognised as a task by OGGM and used to launch snowslide 
    through a workflow using OGGM. This allows snowslide to use all the possibilities offered by OGGM.  
    (see after)

## Snowslide Features

Through a main function defined in snowslide_main, snowslide uses a number of functions which are
present in the ``functions.py`` file. These are the following : 
- **dem_flow()** : This function can preprocess the dem (or not) and compute the flow direction based on the 
    total elevation surface (dem + snow depth)
- **precipitations_base()** : This function initialize an ideal SND matrix based on solid precipitation information
- **slope()** : This function calculate a slope matrix based on the dem matrix
- **snow_routing()** : This function routes the snow based on the routing method chosen ('mfd' or 'd8'). 
    It is called at each iteration. 
- **SND_max_exponential()**: Function that compute the maximal height of snow each pixel can store based on the slope. 
    The function is an exponential and parameters are estimated from 'Bernhardt & Schulz 2007'.
- **reframe_tif()** : To quickly crop a DEM, retaining only the area of interest. It displays the initial DEM, 
    and the user can then enter the x and y windows of the zone to be retained so that the function can store 
    a new DEM entitled: 'reframed_dem'.
- **resampling_dem()** : This function allows to resample a dem to another resolution. It can be used to increase 
    the speed of calculations if high resolution is not required.

## Getting to grips with the Snowslide environment

To make it easier for users to use the package, we have produced a series of tutorial notebooks that explain how to use snowslide in conjunction with OGGM. These notebooks fall into 3 categories:

## How to apply SnowSlide on a DEM to obtain distributed avalanche correction factors

Check out this [notebook](https://github.com/OGGM/Snowslide/blob/main/notebooks/avalanche_maps_for_gdirs.ipynb)

Note that you can easily adapt this notebook to work with your own DEM, independently from OGGM.

## How to use SnowSlide in conjunction with OGGM

If you intend to use the distributed avalanche correction factors to calibrate your mass balance model in OGGM
check out this [notebook](https://github.com/OGGM/Snowslide/blob/main/notebooks/running_oggm_with_avalanches.ipynb)
