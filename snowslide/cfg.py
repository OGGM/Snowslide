"""This configuration module is a container for parameters and constants."""
import numpy as np

# This is a parameter container
# It is a dictionary because we want it to be updated at runtime
PARAMS = dict()


def set_default_params():
    """Initializes PARAMS with the default values"""

    global PARAMS
    PARAMS['snow_density'] = 150 # kg/m3
    PARAMS['a'] = - 0.14 # exponential factor
    PARAMS['c'] = 145  # exponential factor
    PARAMS['pi'] = np.pi 
    PARAMS['epsilon'] = 1e-3  # endloop condition

# Make sure they are set at first import
set_default_params()
