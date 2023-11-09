# This is hard coded version string.
# Real packages use more sophisticated methods to make sure that this
# string is synchronised with `setup.py`, but for our purposes this is OK
__version__ = '0.0.1'

# __init__ is often used as entry point for what is called "the API"
# (application programming interface). This way, users of the lib don't
# have to know about the internal structure of a package, they just want
# to know what functionality it provides

from snowslide.functions import dem_flow 
from snowslide.functions import precipitations
from snowslide.functions import precipitations_base
from snowslide.functions import snd_max_exponential
from snowslide.functions import snow_routing
from snowslide.functions import slope
from snowslide.functions import reframe_tif
from snowslide.functions import resampling_dem

from snowslide.snowslide_main import snowslide_base
from snowslide.snowslide_main import snowslide_complete



