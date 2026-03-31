from snowslide.functions import *

# Module logger
import logging

log = logging.getLogger(__name__)


def snowslide_base(
    path_dem,
    snd0,
    save_path=None,
    epsilon=1e-3,
    param_expo={"a": 0.14, "c": 145, "min": 0.05},
    param_routing={"routing": "mfd", "preprocessing": True, "compute_edges": True},
    glacier_id="",
    propagation_boolean=True
):
    """This function operates the gravitationnal transport of the snow from an initial map of snow heights and a dem.
    Snowslide_base is the fastest computing snowslide algorithm with very basic display functionalities allowed for the user.

    Parameters
    ----------
    path_dem: str
        Path to the dem file (.tif)
    snd0: numpy.ndarray
        Numpy matrix of the same size as the dem containing the initial snow depths (derived from precipitation).
    save_path: str
        Path where the user want to save the data produced (.tif file of the snd matrix). Default is None.
        If no path is indicated, no data is stored (only registered as a variable when running the function)
        If a path is indicated the matrix will be saved as 'converged_SND.tif' file
    epsilon: float
        Condition to get out the loop (convergence is considered reached when indicateur < epsilon). Default is 1e-3.
    param_expo: dictionary
        The maximum snow height each pixel can store is computed as an exponential function of the slope. Parameters are the following:
            snd_max=c*exp(a*slope)
        a: float
            Default is 0.14.
        c: float
            Default is 145
        min: float
            Default is 0.05.
    param_routing: dictionary
        Parameters used by pysheds to realize the routing of the snow.
        routing: str
            routing method ('mfd' or 'd8')
        preprocessing: bolean
            activate or deactivate preprocessing of the DEM. Deactivate it affects convergence.
        compute_edges: bolean
            activate or deactivate computing of edges for the slope function
    glacier_id : str, optional
        for logging only: add the name of the glacier to the log messages
    propagation_boolean : boolean
        for reduction of snow holding depth for pixels receiving avalanches (as in Queno et al., 2023: https://doi.org/10.5194/egusphere-2023-2071)

    Returns
    -------
    Final matrix of the snow height (np matrix in meters). It is stored in the folder if indicated by the user
    """

    ### Initialization ###
    iter = 0
    indicateur = float("inf")

    # useful dem information for registering output as a .tif file
    get_info = rasterio.open(path_dem)
    crs = str(get_info.crs)
    resolution = float(get_info.res[0])
    transform = get_info.transform

    # Importation of dem using pysheds
    grid = Grid.from_raster(path_dem)
    dem = grid.read_raster(path_dem)

    # initialization of initial snow depths
    snd = np.copy(snd0)

    log.debug(
        f"{glacier_id}variables have been initialized, launching the simulation..."
    )

    ### Core part of the code ###

    convergence = []
    while indicateur > epsilon:
        snd1 = np.copy(snd)
        hnso = (
            dem + snd
            ) # total elevation surface (HNSO) is recalculated at each iteration
        flow_dir = dem_flow(
            hnso, grid, param_routing["routing"], param_routing["preprocessing"]
        )
        # Force edges to zero to prevent "infinite snow" from boundary pixels (to adapt to pysheds 0.5 version - not needed for v0.3.5)
        # FORCE EDGES TO ZERO
        # This prevents center pixels from "pulling" snow from the static boundary
        if param_routing["routing"] == "mfd":
            flow_dir[:, 0, :] = 0   # Top
            flow_dir[:, -1, :] = 0  # Bottom
            flow_dir[:, :, 0] = 0   # Left
            flow_dir[:, :, -1] = 0  # Right
        else: # For D8
            flow_dir[0, :] = 0   # Top row
            flow_dir[-1, :] = 0  # Bottom row
            flow_dir[:, 0] = 0   # Left col
            flow_dir[:, -1] = 0  # Right col
        hnso_slope = slope(hnso, resolution, resolution, param_routing["compute_edges"])
        snd_max = snd_max_exponential(
            hnso_slope, param_expo["a"], param_expo["c"], param_expo["min"]
        )
        # reduce snow depth threshold by 30% for pixels that have revived snow, i.e. pixels tht have more snow than snd_max after the first iteration
        snd_flat = snd.flatten()
        snd_max_flat = snd_max.flatten()
        if (iter > 0) & propagation_boolean:
            snd_max_flat[snd_flat>snd_max_flat] = 0.70*snd_max_flat[snd_flat>snd_max_flat]
        snd_max_updated = snd_max_flat.reshape(snd_max.shape)
        snd_max = snd_max_updated

        snd = snow_routing(snd, snd_max, flow_dir, param_routing["routing"])

        iter = iter + 1

        ### Exit conditions ###

        # 1st exit condition
        indicateur = np.sum(
            (snd - snd1) ** 2
        )  # Exit condition when the L2 norm of the distance
        # between the matrices of two iterations converge towards 0.
        convergence.append(indicateur)

        # 2nd exit condition if indicateur converge towards a constant that is not 0.
        if iter > 5:
            speed = np.sum(np.gradient(np.array(convergence)[-5:]))
            if speed < 0 and abs(speed) < epsilon:
                indicateur = 0

        # 3rd exit condition if more than 500 iterations.
        if iter > 500:
            indicateur = 0

    log.info(f"{glacier_id}the algorithm converged in {iter} iterations")

    # Saving the output as .tif file with same projection and properties as the dem
    if save_path != None:
        output_path = save_path + "/converged_SND.tif"
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=snd.shape[0],
            width=snd.shape[1],
            count=1,
            dtype=snd.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(snd, 1)

        log.debug(
            f"{glacier_id}the file has been stored in the following location: {output_path}"
        )

    return snd

