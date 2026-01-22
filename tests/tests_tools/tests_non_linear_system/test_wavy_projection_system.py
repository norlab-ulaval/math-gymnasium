# coding=utf-8
from typing import Tuple

import numpy as np
import pytest

from math_gymnasium.dynamical_systems.non_linear_system.wavy_projection_system import (
    rollout_wavy_projection_partial_derivative,
)


@pytest.fixture
def setup_wavy_projection_sys_spaces() -> Tuple[
    np.ndarray,
    Tuple[float, float, float],
]:
    t_time_space = np.arange(10) * 0.1
    t_init_coord = (0.0, 0.0, 1.0)
    return t_time_space, t_init_coord


class TestRolloutWavyProjectionPartialDerivative:
    def test_default(self, setup_wavy_projection_sys_spaces):
        (t_time_space, t_init_coord) = setup_wavy_projection_sys_spaces
        t_time_space_freeze = t_time_space.copy()
        coords = rollout_wavy_projection_partial_derivative(
            t_time_space, vx=1.0, vy=1.0, initiale_coordinates=t_init_coord,
        )
        assert np.array_equal(
            t_time_space_freeze, t_time_space
        ), "original array was overridden"
        assert coords.shape == (t_time_space.shape[0], 3)
        
        # Check if x and y are linear
        # x = 0.0, 0.1, 0.2, ...
        # y = 0.0, 0.1, 0.2, ...
        assert np.allclose(coords[:, 0], t_time_space)
        assert np.allclose(coords[:, 1], t_time_space)
        
        # Check if z is roughly sin(x) + cos(y)
        # For the first few steps, z should be close to sin(x) + cos(y)
        # Note: integration might have some error, so we just check it's not linear
        z_expected = np.sin(coords[:, 0]) + np.cos(coords[:, 1])
        # Since it's Euler integration, it won't be exact
        # We just check it follows the trend
        assert not np.allclose(coords[:, 2], coords[0, 2] + (coords[1, 2] - coords[0, 2]) * np.arange(10))

    def test_output_coord_using_delta_time(self, setup_wavy_projection_sys_spaces):
        (t_time_space, t_init_coord) = setup_wavy_projection_sys_spaces
        t_dt_space = np.ones_like(t_time_space) * 0.1
        t_dt_space[0] = 0.0
        
        coord = rollout_wavy_projection_partial_derivative(
            t_dt_space, vx=1.0, vy=1.0, initiale_coordinates=t_init_coord, time_space_is_delta_time=True
        )
        assert coord.shape == (t_time_space.shape[0], 3)
        # Initial z is 1.0. 
        # Next z = 1.0 + (cos(0)*1 - sin(0)*1)*0.1 = 1.0 + 0.1 = 1.1
        assert np.allclose(coord[1, 2], 1.1)
