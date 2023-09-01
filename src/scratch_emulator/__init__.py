import hashlib
import importlib.metadata
import itertools
from typing import Callable, Literal

import numpy as np
import rtm_wrapper.simulation as rtm_sim
import scipy.spatial.distance as sci_dist
import xarray as xr

__version__ = importlib.metadata.version("scratch-emulator")

from rtm_wrapper.simulation import SweepSimulation


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


def sweep_hash(sweep: SweepSimulation) -> str:
    """
    Compute hash identifying a particular sweep. Not guaranteed to be stable
    between versions.
    """
    h = hashlib.new("sha256")
    for name, coord in sweep.sweep_spec.coords.items():
        h.update(coord.values.tobytes())
    h.update(repr(sweep.base).encode("ascii"))

    return h.hexdigest()


def brute_maximin(
    num_samples: int,
    dims: int,
    iterations: int = 1000,
    pick: Literal["random", "min"] = "min",
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "euclidean",
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    samples = rng.uniform(size=(num_samples, dims))

    all_dists = sci_dist.cdist(samples, samples, metric)
    np.fill_diagonal(all_dists, np.inf)
    min_dist = all_dists.min()

    for _ in itertools.repeat(None, iterations):
        if pick == "min":
            i_row = np.unravel_index(np.argmin(all_dists), all_dists.shape)[0]
        else:
            i_row = rng.integers(0, num_samples)

        old_row = samples[i_row, :]
        samples[i_row, :] = rng.uniform(size=dims)

        new_dists = sci_dist.cdist(samples, samples, metric)
        np.fill_diagonal(new_dists, np.inf)

        if (d := new_dists.min()) > min_dist:
            min_dist = d
            all_dists = new_dists
        else:
            samples[i_row, :] = old_row

    return samples
