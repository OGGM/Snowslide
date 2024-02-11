# A simple parametrisation of the transport and deposition of snow on steep terrains.

The **snowslide** python package was written by Guillem Carcanade, intern in 
the glaciology team of IGE Grenoble based on the algorithms described by 
Bernhardt & Schulz, 2010 and Gruber, 2007.  

This package provides a simple parametrisation of the transport and deposition 
of snow in mountain terrains. It has been designed for applications in 
glaciological modeling, and specifically to be used through the OGGM workflow. 
While the package provides methods for use in conjunction with OGGM, the basic 
functions can also be called up and used independently. Snowslide can then work 
in a wide variety of context that includes gravitational transport. Snowslide operates the estimated redistribution 
of snow by avalanches using only two inputs : 
- a dem given as a path to a .tif file
- information about initial snow depths 

The next steps goes into more details about how snowslide works and should be used.

## Installation 

Snowslide relies on basic python librairies for data manipulation. Nonetheless, 
the routing of the snow is calculated by an external python module called pysheds
(docs : https://github.com/mdbartos/pysheds/). Running snowslide therefore requires
installing pysheds (pip install pysheds).

The list of all snowslide dependancies is given as follow : 
- pysheds
- rasterio
- numpy
- xarray
- pandas
- matplotlib
- os
- datetime

These libraries must be installed to run snowslide. Then you can install 
snowslide using pip : 

    $ pip install -e git+https://github.com/GuillemCF/Snowslide.git

This should clone the snowslide GitHub repository and install it as a python
package in the active virtual environment chosen. 

## Package structure

#### Directory root (``./``)

- ``.gitignore``: for git users only
- ``LICENSE.txt``: (https://help.github.com/articles/licensing-a-epository/) license of the code
- ``README.md``: this page
- ``pyproject.toml``: this is what makes your package insallable by ``pip``. It
  contains a set of simple instructions regarding e.g. the name of the package,
  its version number, or where to find command line scripts.
  
#### The actual package (``./snowslide``)

- ``__init__.py``: tells python that the directory is a package and enables
  the  "dotted module names"  import syntax. It is often empty,fla but here
  we added some entry points to the package's API and the version string.
- ``snowslide_main.py``: main module that operates the simulation
- ``functions.py``: various functions used in the main module and defined separatly 
- ``display_2Dresults.py``: very basic display functionalities ofered to the user
- ``display_3Dresults.py``: very basic display functionalities ofered to the user (Disabled for now)
- ``oggm_snowslide_compat.py``: module recognised as a task by OGGM and used to launch snowslide 
    through a workflow using OGGM. This allows snowslide to use all the possibilities offered by OGGM.  
    (see after)

#### The tests (``./snowslide/test``)

``test_snowslide.py``: is a module that can be detected by pytest. It includes various functions
that test the expected behaviour of snowslide algorithm.
``./data`` : contains basic data to run the tests

## Snowslide Features

Through a main function defined in snowslide_main, snowslide uses a number of functions which are
present in the ``functions.py`` file. These are the following : 
- **dem_flow()** : This function can preprocess the dem (or not) and compute the flow direction based on the 
    total elevation surface (dem + snow depth)
- **precipitations_base()** : This function initialize an ideal SND matrix based on solid precipitation information
- **slope()** : This function calculate a slope matrix based on the dem matrix
- **snow_routing()** : That function routes the snow based on the routing method chosen ('mfd' or 'd8'). 
    It is called at each iteration. 
- **SND_max_exponential()**: Function that compute the maximal height of snow each pixel can store based on the slope. 
    The function is an exponential and parameters are estimated from 'Bernhardt & Schulz 2007'.
- **reframe_tif()** : To quickly crop a DEM, retaining only the area of interest. It displays the initial DEM, 
    and the user can then enter the x and y windows of the zone to be retained so that the function can store 
    a new DEM entitled: 'reframed_dem'.
- **resampling_dem()** : This function allows to resample a dem to another resolution. It can be used to increase 
    the speed of calculations if high resolution is not required.

The ``display_2Dresults.py`` and ``display_3Dresults.py`` are used in snowslide_complete() function defined in 
``snowslide_main.py``. This function offers the same simulation as snowslide_base(), but simply provides functions 
for recording and displaying output that can simplify the use of snowslide for the user in certain cases. We don't 
use it at the moment, so we recommend that users use the snowslide_base() function instead. The rest of the readme 
gives an example of using snowslide first in a snowslide workflow and then in an OGGM workflow.

## Getting to grips with the snowslide environment

To make it easier for users to use the package, we have produced a series of tutorial notebooks that explain how to use snowslide, its principles and its performance. These notebooks fall into 3 categories:

## How to use the package 

To understand how to use snowslide in a classic workflow, see the notebook in https://github.com/GuillemCF/Snowslide/blob/main/example/How_to_use_snowslide.ipynb
To understand how to use snowslide in a OGGM workflow, see the notebook in https://github.com/GuillemCF/Snowslide/blob/main/example/How_to_use_OGGMxSnowslide.ipynb
To understand how to use snowslide output from the OGGM workflow, see the notebook in https://github.com/OGGM/Snowslide/blob/main/example/How_to_use_OGGMxSnowslidePrePro.ipynb
To get to grips with the preprocessing functions associated with the algorithm, see the notebook ...

## How is the model created? 

To understand how snow heights can be initialized in the model, see the notebook: ...
To understand how snow is routed and how convergence is ensured, see the notebook: ...

## Analysis of various results and study of the algorithm's performance

To see the results of the algorithm under idealized conditions, see the notebook: ...
To see the results of the algorithm in real-life conditions, see the notebook: ...
To see the results of the algorithm applied globally coupled with OGGM, see the notebook: ...
To analyse the algorithm's computational performance, see the notebook: ...
To understand the limit cases for using the algorithm, see the notebook: ...