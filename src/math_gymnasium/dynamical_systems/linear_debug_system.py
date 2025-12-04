# coding=utf-8

import numpy as np

def rollout_linear_debug_system(
    time_space: np.ndarray,
    initiale_coordinates=(0.0, 0.0, 0.0),
    time_space_is_delta_time: bool = False,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Calculates the partial derivatives of a linear debugging system over a given time space.

    Observation partern:

        - x: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, ...
        - y: 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, ...
        - z: 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, ...
        - act: 1, 2, 3, 4, 5, 6, ...

    Assume `time_space` values are increassing if `time_space_is_delta_time=True`

    :param time_space: Array representing time steps in wallclock time or delta time.
    :param initiale_coordinates: The state at timestep 0
    :param time_space_is_delta_time: Param not used. Only there for matching signature.
    :param dtype: Data type for computations.
    :return: Array of computed x, y, z coordinates over the given time space.
    """
    assert isinstance(initiale_coordinates, tuple) and len(initiale_coordinates) == 3
    assert isinstance(time_space, np.ndarray) and time_space.ndim == 1

    xyzs = np.tile(np.arange(time_space.size, dtype=dtype), 3)
    xyzs = xyzs.reshape(3, -1).swapaxes(-2, -1)

    xyzs += initiale_coordinates

    x_dot = 0.1
    y_dot = 0.2
    z_dot = 0.3
    xyz_dot = np.array([x_dot, y_dot, z_dot], dtype=dtype)

    for i in np.arange(time_space.size):
        xyzs[i] += xyz_dot

    return xyzs
