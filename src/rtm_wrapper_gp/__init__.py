import importlib.metadata

import numpy as np
import rtm_wrapper.simulation as rtm_sim
import torch
import xarray as xr

__version__ = importlib.metadata.version("rtm_wrapper_gp")


def dataarray2tensors(
    data: xr.DataArray,
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    input_coords = [
        coord.name
        for dim in data.dims
        for coord in data.coords.values()
        if coord.name.partition("__")[0] in rtm_sim.INPUT_TOP_NAMES
        and coord.dims == (dim,)
    ]
    input_coords.sort()

    input_points = np.empty((data.size, len(input_coords)))

    for row, dim_index in enumerate(np.ndindex(data.shape)):
        point = data[dim_index]

        input_points[row, :] = tuple(
            getattr(point, coord).item() for coord in input_coords  # type: ignore
        )

    x = torch.tensor(input_points)
    y = torch.tensor(data.values.ravel())

    return tuple(input_coords), x, y


def unit2range(arr: np.ndarray, bot: float, top: float) -> np.ndarray:
    return arr * (top - bot) + bot
