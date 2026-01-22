# coding=utf-8
from typing import Tuple

import numpy as np
import pytest

from math_gymnasium.dynamical_systems.non_linear_system.sombrero_projection_system import (
    rollout_sombrero_projection_partial_derivative,
)


@pytest.fixture
def setup_sombrero_projection_sys_spaces() -> Tuple[
    np.ndarray,
    Tuple[float, float, float],
]:
    t_time_space = np.arange(10) * 0.1
    t_init_coord = (0.0, 0.0, 1.0)
    return t_time_space, t_init_coord


class TestRolloutSombreroProjectionPartialDerivative:
    def test_default(self, setup_sombrero_projection_sys_spaces):
        (t_time_space, t_init_coord) = setup_sombrero_projection_sys_spaces
        t_time_space_freeze = t_time_space.copy()
        coords = rollout_sombrero_projection_partial_derivative(
            t_time_space, vx=1.0, vy=1.0, initiale_coordinates=t_init_coord,
        )
        assert np.array_equal(
            t_time_space_freeze, t_time_space
        ), "original array was overridden"
        assert coords.shape == (t_time_space.shape[0], 3)
        
        # Check if x and y are linear
        assert np.allclose(coords[:, 0], t_time_space)
        assert np.allclose(coords[:, 1], t_time_space)
        
        # Initial z should be A = 1.0
        assert np.allclose(coords[0, 2], 1.0)
        
        # Check that it's not linear
        assert not np.allclose(coords[:, 2], coords[0, 2] + (coords[1, 2] - coords[0, 2]) * np.arange(10))

    def test_output_coord_using_delta_time(self, setup_sombrero_projection_sys_spaces):
        (t_time_space, t_init_coord) = setup_sombrero_projection_sys_spaces
        t_dt_space = np.ones_like(t_time_space) * 0.1
        t_dt_space[0] = 0.0
        
        coord = rollout_sombrero_projection_partial_derivative(
            t_dt_space, vx=1.0, vy=1.0, initiale_coordinates=t_init_coord, time_space_is_delta_time=True
        )
        assert coord.shape == (t_time_space.shape[0], 3)
        # Initial z is 1.0.
        # At r=0, dz/dt = 0 (as calculated in thoughts, since x*vx+y*vy = 0).
        # Actually, let's check the first step from origin.
        # If x=0, y=0, then x*vx + y*vy = 0, so dz/dt = 0.
        # So the second z should still be 1.0 if starting EXACTLY at origin.
        assert np.allclose(coord[1, 2], 1.0)
        
        # If we start offset, it should change.
        coord_offset = rollout_sombrero_projection_partial_derivative(
            t_dt_space, vx=1.0, vy=1.0, initiale_coordinates=(0.1, 0.1, 1.0), time_space_is_delta_time=True
        )
        assert not np.allclose(coord_offset[1, 2], 1.0)
