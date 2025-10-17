# coding=utf-8

import numpy as np

from tools.math_tools.space_conversion_tools.time_to_delta_time import (
    convert_state_time_to_state_delta_time,
)
from tools.math_tools.ndarray_tools.custom_msg import nan_infinity_console_warning


def lorenz_attractor_partial_derivative(
    xyz: np.ndarray,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 2.667,
    dtype: np.dtype = np.float64,
    debug: bool = False,
):
    """
    Computes the partial derivatives for the Lorenz attractor system chaotic time series.

    Inspired by: https://en.wikipedia.org/wiki/Lorenz_system#Python_simulation

    :param xyz: An array containing the x, y, and z coordinates.
    :param s: Initial condition Sigma parameter of the Lorenz system.
    :param r: Initial condition Rho parameter of the Lorenz system.
    :param b: Initial condition Beta parameter of the Lorenz system.
    :param dtype:
    :param debug:
    :return: An array containing the partial derivatives [x_dot, y_dot, z_dot].
    """

    if debug and not np.all(np.isfinite(xyz)):
        nan_infinity_console_warning("xyz")

    # Note: if coordinate value are not finite, then coordinate have drifted to far
    #       and it's ok to handle them via nan to num fct
    xyz = np.nan_to_num(xyz)

    x, y, z = xyz.astype(dtype)
    s = np.array(s, dtype)
    r = np.array(r, dtype)
    b = np.array(b, dtype)

    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    xyz_dot = np.array([x_dot, y_dot, z_dot], dtype=dtype)

    if debug and not np.all(np.isfinite(xyz_dot)):
        nan_infinity_console_warning("xyz_dot")

    xyz_dot = np.nan_to_num(xyz_dot)
    return xyz_dot.squeeze()


def rollout_lorenz_attractor_partial_derivative(
    time_space: np.ndarray,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 2.667,
    initiale_coordinates=(0.0, 1.0, 1.05),
    time_space_is_delta_time: bool = False,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Calculates the partial derivatives of the Lorenz attractor equations over a given time space.

    Assume `time_space` values are increassing if `time_space_is_delta_time=True`

    :param time_space: Array representing time steps in wallclock time or delta time.
    :param s: Initial condition Sigma parameter of the Lorenz system.
    :param r: Initial condition Rho parameter of the Lorenz system.
    :param b: Initial condition Beta parameter of the Lorenz system.
    :param initiale_coordinates: The state at timestep 0
    :param time_space_is_delta_time: Set to True if `time_space` is an array of delta time.
    :param dtype:
    :return: Array of computed x, y, z coordinates over the given time space.
    """
    assert isinstance(initiale_coordinates, tuple) and len(initiale_coordinates) == 3
    assert isinstance(time_space, np.ndarray) and time_space.ndim == 1

    xyzs = np.empty((time_space.size, 3), dtype=dtype)
    xyzs_partial_derivative = np.empty_like(xyzs)

    xyzs[0] = initiale_coordinates
    xyzs_partial_derivative[0] = (0.0, 0.0, 0.0)
    if not time_space_is_delta_time:
        delta_time = convert_state_time_to_state_delta_time(time_space.astype(dtype))
    else:
        delta_time = time_space.copy().astype(dtype)

    for i in np.arange(time_space.size - 1):
        xyzs_partial_derivative[i + 1] = lorenz_attractor_partial_derivative(
            xyzs[i], s, r, b, dtype
        )
        xyzs[i + 1] = xyzs[i] + xyzs_partial_derivative[i + 1] * delta_time[i + 1]

    assert np.all(
        np.isfinite(xyzs_partial_derivative)
    ), "Nan value detected in `xyzs_partial_derivative`"
    assert np.all(np.isfinite(xyzs)), "Nan value detected in `xyzs`"
    return xyzs
