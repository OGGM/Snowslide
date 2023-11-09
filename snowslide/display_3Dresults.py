# Librairies imports
import os
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
import rasterio


def save_plots3D(
    path_dem,
    SND_plot=None,
    save_path=None,
    param_camera={
        "factor": 10,
        "cellsize": 30,
        "azimuth": 0,
        "elevation": 45,
        "distance": "auto",
    },
):
    """This function allows to plot and save a 3D representation of a dem superimposed with a snow depth matrix
    For easier viewing, snow heights are multiplied by a factor chosen from the param_camera dictionary.

    Parameters
    ----------
    path_dem: str
        Path to the DEM used in the simulation
    SND_plot: np matrix
        Matrix of the snow height
    save_path: where to save the images
    param_camera: dictionary
        Settings that allow the user to control the 3D display offered by mayavi
        param_camera['factor']: float, multiplication of snow heights to make them easier to see. Default is 10.
        param_camera['cellsize']: float, resolution of the DEM in meters. Default is 30.
        param_camera['azimuth']: 0:360, horizontal viewing angle in degrees. Default is 0.
        param_camera['elevation']: -90:90, vertical viewing angle in degrees. Default is 45.
        param_camera['distance']: float, distance of the camera to the object. Default is 'auto'.
    More information about viewing parameters can be find in the following mayavi documentation:
    'https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html'

    Returns
    -------
    The function stores a 3D image representation of the modeled snow heights.
    """

    with rasterio.open(path_dem) as ds:
        dem = ds.read(1)

        if (
            SND_plot == None
        ):  # If the user does not enter a SND matrix it then shows only the dem
            SND_plot = np.zeros(np.shape(dem))

        HNSO_plot = dem + SND_plot * param_camera["factor"]

        # Plot DEM + SND with mayavi
        mlab.options.offscreen = True
        mlab.figure(size=(640, 800), bgcolor=((0.16, 0.28, 0.46)))
        mlab.surf(dem, warp_scale=(1 / param_camera["cellsize"]), colormap="copper")
        mlab.surf(
            HNSO_plot,
            warp_scale=(1 / param_camera["cellsize"]),
            colormap="PuBu",
            opacity=0.9,
        )

        # Camera settings
        azimuth = param_camera["azimuth"]  # contrôle l'angle horizontal de l'image
        elevation = param_camera["elevation"]  # contrôle l'angle vertical de l'image
        distance = param_camera[
            "distance"
        ]  # contrôle la distance à la figure, 'auto' laisse mayavi faire le choix

        view = mlab.view()
        mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)

        # Storing the representation as an image
        if save_path == None:
            folder_path = os.getcwd()
            save_path = folder_path + "/3D_SNDimage.png"
        mlab.savefig(save_path)
