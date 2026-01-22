# coding=utf-8

import numpy as np

from tools.math_tools.ndarray_tools.custom_msg import nan_infinity_console_warning
from tools.math_tools.space_conversion_tools.time_to_delta_time import (
    convert_state_time_to_state_delta_time,
)


def wavy_projection_partial_derivative(
    xyz: np.ndarray,
    vx: float = 1.0,
    vy: float = 1.0,
    dtype: np.dtype = np.float64,
    debug: bool = False,
):
    """
    Computes the partial derivatives for a wavy projection system.
    This is a 2D linear motion in (x, y) projected onto a 3D sinusoidal surface.

    System equations:
    dx/dt = vx
    dy/dt = vy
    dz/dt = cos(x)*vx - sin(y)*vy  =>  (Integration: z = sin(x) + cos(y) + C)

    :param xyz: An array containing the x, y, and z coordinates.
    :param vx: Velocity in x direction.
    :param vy: Velocity in y direction.
    :param dtype: Data type for computations.
    :param debug: Warn if nan or infinity values are encountered.
    :return: An array containing the partial derivatives [x_dot, y_dot, z_dot].
    """
    xyz = np.nan_to_num(xyz)
    x, y, z = xyz.astype(dtype)

    x_dot = vx
    y_dot = vy
    # z is the projection on sin(x) + cos(y)
    # dz/dt = d(sin(x) + cos(y))/dt = cos(x)*dx/dt - sin(y)*dy/dt
    z_dot = np.cos(x) * vx - np.sin(y) * vy

    xyz_dot = np.array([x_dot, y_dot, z_dot], dtype=dtype)

    if debug and not np.all(np.isfinite(xyz_dot)):
        nan_infinity_console_warning("xyz_dot")

    xyz_dot = np.nan_to_num(xyz_dot)
    return xyz_dot.squeeze()


def rollout_wavy_projection_partial_derivative(
    time_space: np.ndarray,
    vx: float = 1.0,
    vy: float = 1.0,
    initiale_coordinates=(0.0, 0.0, 1.0), # z = sin(0) + cos(0) = 1.0
    time_space_is_delta_time: bool = False,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Calculates the trajectory of a wavy projection system over a given time space.

    Assume `time_space` values are increasing if `time_space_is_delta_time=True`

    :param time_space: Array representing time steps in wallclock time or delta time.
    :param vx: Velocity in x direction.
    :param vy: Velocity in y direction.
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
        xyzs_partial_derivative[i + 1] = wavy_projection_partial_derivative(
            xyzs[i], vx, vy, dtype
        )
        xyzs[i + 1] = xyzs[i] + xyzs_partial_derivative[i + 1] * delta_time[i + 1]

    assert np.all(np.isfinite(xyzs_partial_derivative))
    assert np.all(np.isfinite(xyzs))
    return xyzs
