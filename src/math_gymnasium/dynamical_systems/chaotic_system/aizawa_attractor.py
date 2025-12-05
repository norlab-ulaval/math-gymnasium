# coding=utf-8

import numpy as np

from tools.math_tools.space_conversion_tools.time_to_delta_time import (
    convert_state_time_to_state_delta_time,
)
from tools.math_tools.ndarray_tools.custom_msg import nan_infinity_console_warning

# (CRITICAL) ToDo: unit and integration tests (ref task MG-37)

def aizawa_attractor_partial_derivative(
    xyz: np.ndarray,
    a: float = 0.95,
    b: float = 0.7,
    c: float = 0.6,
    d: float = 3.5,
    e: float = 0.25,
    f: float = 0.1,
    dtype: np.dtype = np.float64,
    debug: bool = False,
):
    """
    Computes the partial derivatives for the Aizawa attractor system.

    System equations:
    dx/dt = (z - b)*x - d*y
    dy/dt = d*x + (z - b)*y
    dz/dt = c + a*z - z^3/3 - (x^2 + y^2)*(1 + e*z) + f*z*x^3

    :param xyz: An array containing the x, y, and z coordinates.
    :param a: System parameter (typically 0.95).
    :param b: System parameter (typically 0.7).
    :param c: System parameter (typically 0.6).
    :param d: System parameter (typically 3.5).
    :param e: System parameter (typically 0.25).
    :param f: System parameter (typically 0.1).
    :param dtype: Data type for computations.
    :param debug: Enable debug mode.
    :return: An array containing the partial derivatives [x_dot, y_dot, z_dot].
    """
    if debug and not np.all(np.isfinite(xyz)):
        nan_infinity_console_warning("xyz")

    # Note: if coordinate value are not finite, then coordinate have drifted to far
    #       and it's ok to handle them via nan to num fct
    xyz = np.nan_to_num(xyz)

    x, y, z = xyz.astype(dtype)
    a = np.array(a, dtype)
    b = np.array(b, dtype)
    c = np.array(c, dtype)
    d = np.array(d, dtype)
    e = np.array(e, dtype)
    f = np.array(f, dtype)

    x_dot = (z - b) * x - d * y
    y_dot = d * x + (z - b) * y
    z_dot = c + a * z - (z**3) / 3.0 - (x**2 + y**2) * (1.0 + e * z) + f * z * (x**3)

    xyz_dot = np.array([x_dot, y_dot, z_dot], dtype=dtype)

    if debug and not np.all(np.isfinite(xyz_dot)):
        nan_infinity_console_warning("xyz_dot")

    xyz_dot = np.nan_to_num(xyz_dot)
    return xyz_dot.squeeze()


def rollout_aizawa_attractor_partial_derivative(
    time_space: np.ndarray,
    a: float = 0.95,
    b: float = 0.7,
    c: float = 0.6,
    d: float = 3.5,
    e: float = 0.25,
    f: float = 0.1,
    initiale_coordinates=(0.1, 0.0, 0.0),
    time_space_is_delta_time: bool = False,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Calculates the trajectory of the Aizawa attractor over a given time space.

    Assume `time_space` values are increassing if `time_space_is_delta_time=True`

    :param time_space: Array representing time steps in wallclock time or delta time.
    :param a: System parameter (typically 0.95).
    :param b: System parameter (typically 0.7).
    :param c: System parameter (typically 0.6).
    :param d: System parameter (typically 3.5).
    :param e: System parameter (typically 0.25).
    :param f: System parameter (typically 0.1).
    :param initiale_coordinates: The state at timestep 0
    :param time_space_is_delta_time: Set to True if `time_space` is an array of delta time.
    :param dtype: Data type for computations.
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
        xyzs_partial_derivative[i + 1] = aizawa_attractor_partial_derivative(
            xyzs[i], a, b, c, d, e, f, dtype
        )
        xyzs[i + 1] = xyzs[i] + xyzs_partial_derivative[i + 1] * delta_time[i + 1]

    assert np.all(np.isfinite(xyzs_partial_derivative))
    assert np.all(np.isfinite(xyzs))
    return xyzs
