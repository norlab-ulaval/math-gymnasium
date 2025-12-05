# coding=utf-8

import numpy as np

from tools.math_tools.ndarray_tools.custom_msg import nan_infinity_console_warning
from tools.math_tools.space_conversion_tools.time_to_delta_time import (
    convert_state_time_to_state_delta_time,
)

# (CRITICAL) ToDo: unit and integration tests (ref task MG-42)

def simple_limit_cycle_partial_derivative(
    xyz: np.ndarray,
    mu: float = 1.0,
    omega: float = 1.0,
    alpha: float = 0.2,
    dtype: np.dtype = np.float64,
    debug: bool = False,
):
    """
    Computes the partial derivatives for a simple limit cycle system (Van der Pol-like).

    System equations (simplified Van der Pol):
    dx/dt = y
    dy/dt = mu*(1 - x^2)*y - x
    dz/dt = alpha*(omega - z)

    :param xyz: An array containing the x, y, and z coordinates.
    :param mu: Nonlinearity parameter.
    :param omega: Target z value for relaxation.
    :param alpha: Relaxation rate in z direction.
    :param dtype: Data type for computations.
    :param debug: Warn if nan or infinity values are encountered.
    :return: An array containing the partial derivatives [x_dot, y_dot, z_dot].
    """
    xyz = np.nan_to_num(xyz)
    x, y, z = xyz.astype(dtype)

    x_dot = y
    y_dot = mu * (1.0 - x**2) * y - x
    z_dot = alpha * (omega - z)

    xyz_dot = np.array([x_dot, y_dot, z_dot], dtype=dtype)

    if debug and not np.all(np.isfinite(xyz_dot)):
        nan_infinity_console_warning("xyz_dot")

    xyz_dot = np.nan_to_num(xyz_dot)
    return xyz_dot.squeeze()


def rollout_simple_limit_cycle_partial_derivative(
    time_space: np.ndarray,
    mu: float = 1.0,
    omega: float = 1.0,
    alpha: float = 0.2,
    initiale_coordinates=(0.1, 0.1, 0.0),
    time_space_is_delta_time: bool = False,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Calculates the trajectory of a simple limit cycle system (Van der Pol-like) over a given time space.

    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator

    Assume `time_space` values are increassing if `time_space_is_delta_time=True`

    :param time_space: Array representing time steps in wallclock time or delta time.
    :param mu: Nonlinearity parameter.
    :param omega: Target z value for relaxation.
    :param alpha: Relaxation rate in z direction.
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
        xyzs_partial_derivative[i + 1] = simple_limit_cycle_partial_derivative(
            xyzs[i], mu, omega, alpha, dtype
        )
        xyzs[i + 1] = xyzs[i] + xyzs_partial_derivative[i + 1] * delta_time[i + 1]

    assert np.all(np.isfinite(xyzs_partial_derivative))
    assert np.all(np.isfinite(xyzs))
    return xyzs
