# This is hard coded version string.
# Real packages use more sophisticated methods to make sure that this
# string is synchronised with `setup.py`, but for our purposes this is OK
__version__ = '0.0.1'

# __init__ is often used as entry point for what is called "the API"
# (application programming interface). This way, users of the lib don't
# have to know about the internal structure of a package, they just want
# to know what functionality it provides

from snowslide.utils import dem_flow
from snowslide.utils import precipitations
from snowslide.utils import precipitations_base
from snowslide.utils import SND_max_exponential
from snowslide.utils import snow_routing
from snowslide.utils import slope
from snowslide.utils import reframe_tif
from snowslide.utils import resampling_dem

from snowslide.snowslide import snowslide_base
from snowslide.snowslide import snowslide_complete

#from snowslide.snowslidexOGGM import snowslidexOGGM
#from snowslide.snowslidexOGGM import add_to_flowline



