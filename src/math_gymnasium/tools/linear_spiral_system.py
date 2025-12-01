# coding=utf-8

import numpy as np
from tools.math_tools.space_conversion_tools.time_to_delta_time import (
    convert_state_time_to_state_delta_time,
)


def linear_spiral_partial_derivative(
    xyz: np.ndarray,
    a: float = -0.1,
    omega: float = 1.0,
    c: float = 0.5,
    dtype: np.dtype = np.float64,
    debug: bool = False,
):
    """
    Computes the partial derivatives for a linear spiral system.

    System equations:
    dx/dt = a*x - omega*y
    dy/dt = omega*x + a*y
    dz/dt = c

    :param xyz: An array containing the x, y, and z coordinates.
    :param a: Damping coefficient (negative for spiral inward).
    :param omega: Angular frequency.
    :param c: Constant vertical velocity.
    :param dtype: Data type for computations.
    :param debug: Enable debug mode.
    :return: An array containing the partial derivatives [x_dot, y_dot, z_dot].
    """
    xyz = np.nan_to_num(xyz)
    x, y, z = xyz.astype(dtype)

    x_dot = a * x - omega * y
    y_dot = omega * x + a * y
    z_dot = c

    xyz_dot = np.array([x_dot, y_dot, z_dot], dtype=dtype)
    xyz_dot = np.nan_to_num(xyz_dot)
    return xyz_dot.squeeze()


def rollout_linear_spiral_partial_derivative(
    time_space: np.ndarray,
    a: float = -0.1,
    omega: float = 1.0,
    c: float = 0.5,
    initiale_coordinates=(1.0, 0.0, 0.0),
    time_space_is_delta_time: bool = False,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Calculates the trajectory of a linear spiral system over a given time space.
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
        xyzs_partial_derivative[i + 1] = linear_spiral_partial_derivative(
            xyzs[i], a, omega, c, dtype
        )
        xyzs[i + 1] = xyzs[i] + xyzs_partial_derivative[i + 1] * delta_time[i + 1]

    assert np.all(np.isfinite(xyzs_partial_derivative))
    assert np.all(np.isfinite(xyzs))
    return xyzs
