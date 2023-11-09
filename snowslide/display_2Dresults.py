# Librairies imports
import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import imageio.v2 as iio


def create_giff(
    src_path, dst_path=None, gif_name="gif_animation", fig_names=None, n=None
):
    """This function creates a giff video from a serie of images stored in a folder
    This function is not very automated and a certain number of parameters need to be specified (no need to do more in our case).

    Parameters
    ----------
    src_path: str
        folder where the images are stored
    dst_path: str
        folder where the .gif is to be stored
    gif_name: str
        name of the .gif animation
    fig_names:
        name of the images
    n: int
        number of images used to create the animation

    Returns
    -------
    Saves a .gif animation in the specified folder
    """

    if fig_names == None:
        # get images files paths
        images_names = os.listdir(src_path)
        extensions_images = [".jpg", ".jpeg", ".png"]
        images_paths = [
            os.path.join(src_path, file)
            for file in images_paths
            if os.path.splitext(file)[1].lower() in extensions_images
            and os.path.isfile(os.path.join(src_path, file))
        ]
        images_paths = sorted(images_paths)

        # create gif
        n = len(images_paths)
        images = np.stack(
            [iio.imread(images_paths[i]) for i in range(1, n + 1)], axis=0
        )
        if dst_path == None:
            iio.mimwrite(src_path + "/" + gif_name, images)
        else:
            iio.mimwrite(dst_path + "/" + gif_name, images)

    else:  # if the images are stored with a specific name indicated by user
        images = np.stack(
            [
                iio.imread(src_path + "/" + fig_names + f"{i}.png")
                for i in range(1, n + 1)
            ],
            axis=0,
        )
        iio.mimwrite(dst_path + "/" + gif_name, images)


def save_plots2D(
    file,
    save_path=None,
    legend=None,
    title=None,
    xlabel=None,
    ylabel=None,
    fix_colorbar={"limit": False, "vmin": None, "vmax": None},
):
    """This function saves a plot as an image with specified parameters

    Parameters
    ----------
    file: 2D array
        The file that is shown through the plotting function imshow
    save_path: str
        Path where the user wants the image to be saved
    legend: str
        Legend accompanying the plot
    title: str
        Title of the plot
    xlabel: str
        Legend of x axis
    ylabel:str
        Legend of y axis
    limit_colorbar: dictionary
        This fix the range of the colorbar to make the animation easier to read (with a fixed caption)
        limit_colorbar['limit']: bolean
            Activate limit_colorbar or not
        limit_colorbar['vmin']: float
            minimum value displayed
        limit_colorbar['vmax']: float
            maximum value displayed

    Returns
    -------
    Saves the plot as an image in the specified folder
    """

    fig, ax = plt.subplots(figsize=(15, 10))
    if fix_colorbar["limit"] == True:
        plt.imshow(file, vmin=fix_colorbar["vmin"], vmax=fix_colorbar["vmax"])
    else:
        plt.imshow(file)

    plt.colorbar(label=legend)
    plt.title(title, size=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if save_path == None:
        save_path = (
            os.getcwd()
        )  # Stored in working directory if users does not indicate a saving path

    plt.savefig(save_path)
    plt.close()


def reframe_tif():
    return None
