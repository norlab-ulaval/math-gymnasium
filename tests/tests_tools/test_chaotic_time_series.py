# coding=utf-8
from typing import Tuple

import numpy as np
import pytest

from math_gymnasium.tools.chaotic_time_series import (
    rollout_lorenz_attractor_partial_derivative,
)

@pytest.fixture
def setup_lorenz_sys_spaces() -> (
    Tuple[
        np.ndarray,
        np.ndarray,
        Tuple[float, float, float],
        np.ndarray,
        np.ndarray,
    ]
):
    t_time_space = np.arange(5) * 0.1
    t_dt_space = np.ones(5) * 0.1
    t_dt_space[0] = 0.0
    t_init_coord = (1.0, 1.0, 1.0)
    t_coord = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 3.6, 0.8333],
            [3.6, 5.95667, 0.97105889],
            [5.95667, 15.0914218, 2.85647868],
            [15.0914218, 28.55944553, 11.08411777],
        ]
    )
    t_derivative = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 26.0, -1.667],
            [26.0, 23.5667, 1.3775889],
            [23.5667, 91.347518, 18.85419794],
            [91.347518, 134.68023732, 82.27639084],
        ]
    )
    return t_time_space, t_dt_space, t_init_coord, t_coord, t_derivative


class TestRolloutLorenzAttractorPartialDerivative:
    def test_default(self, setup_lorenz_sys_spaces):
        (t_time_space, t_dt_space, t_init_coord, t_coord, t_derivative) = setup_lorenz_sys_spaces
        t_time_space_freeze = t_time_space.copy()
        coords = rollout_lorenz_attractor_partial_derivative(
            t_time_space, initiale_coordinates=t_init_coord
        )
        assert np.array_equal(t_time_space_freeze, t_time_space), "original array was overiden"
        assert coords.shape == (t_time_space.shape[0], 3)
        assert coords == pytest.approx(t_coord)

    def test_output_coord_using_delta_time(self, setup_lorenz_sys_spaces):
        (t_time_space, t_dt_space, t_init_coord, t_coord, t_derivative) = setup_lorenz_sys_spaces
        t_dt_space_freeze = t_dt_space.copy()
        coord = rollout_lorenz_attractor_partial_derivative(
            t_dt_space, initiale_coordinates=t_init_coord, time_space_is_delta_time=True
        )
        assert np.array_equal(t_dt_space_freeze, t_dt_space), "original array was overiden"
        assert coord.shape == (t_time_space.shape[0], 3)
        assert coord == pytest.approx(t_coord)


