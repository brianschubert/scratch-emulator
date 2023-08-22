import importlib.metadata

import numpy as np
import rtm_wrapper.simulation as rtm_sim
import xarray as xr

__version__ = importlib.metadata.version("scratch-emulator")


def dataarray2xy(
    data: xr.DataArray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    input_coords = [
        coord.name
        for dim in data.dims
        for coord in data.coords.values()
        # TODO tidy sweep coordinate detection.
        if any(coord.name.startswith(top_name) for top_name in rtm_sim.INPUT_TOP_NAMES)
        and coord.dims == (dim,)
    ]
    input_coords.sort()

    input_points = np.empty((data.size, len(input_coords)))

    for row, dim_index in enumerate(np.ndindex(data.shape)):
        point = data[dim_index]

        input_points[row, :] = tuple(
            getattr(point, coord).item() for coord in input_coords  # type: ignore
        )

    x = np.array(input_points)
    y = data.values.reshape(-1, 1)

    return tuple(input_coords), x, y


def unit2range(arr: np.ndarray, bot: float, top: float) -> np.ndarray:
    return arr * (top - bot) + bot
