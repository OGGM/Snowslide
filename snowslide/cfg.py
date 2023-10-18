"""This configuration module is a container for parameters and constants."""
import numpy as np

# This is a parameter container
# It is a dictionary because we want it to be updated at runtime
PARAMS = dict()


def set_default_params():
    """Initializes PARAMS with the default values"""

    global PARAMS
    PARAMS['snow_density'] = 150
    PARAMS['a'] = - 0.14
    PARAMS['c'] = 145
    PARAMS['pi'] = np.pi
    PARAMS['epsilon'] = 1e-3

# Make sure they are set at first import
set_default_params()
