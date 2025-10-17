# coding=utf-8
from functools import partial
from typing import Callable, Tuple, Union

import pytest
import gymnasium as gym
import numpy as np
from omegaconf import omegaconf

from ..general_test_utilities import numpy_array_print_precision_warning

from math_gymnasium.envs.arbitrary_dim_math_continuous import (
    MathContinuousGymnasium,
)
from math_gymnasium.tools.chaotic_time_series import (
    rollout_lorenz_attractor_partial_derivative,
)
from tools.math_tools.space_conversion_tools.coordinate_to_velocity import (
    convert_dt_state_derivatives_to_state_coordinate,
)
from trajectory_container_tools.dataclasses.math_gymnasium_trajectory_dataclass import (
    MathEnvTrajectoryDataclass,
)


class TestMathContinuousGymnasium:
    @pytest.fixture(scope="function")
    def setup_1D_env_cfg(self) -> Tuple[omegaconf.DictConfig, Callable, str]:
        cfg = omegaconf.OmegaConf.create(
            f"""
            time_space:
                bound: [0,{6.25 * np.pi}]
                granularity: 10000
            global_measurement_noise:
                interval: [0,10000]
                magnitude: 0.05
            measurement_noise:
                - interval: [1000,1800]
                  magnitude: 0.1
                - interval: [2200,3550]
                  magnitude: 0.2
            explorable_space: [[1200,2500],[7500,8800]]
            obs_are_dt_derivatives: true
            obs_time_is_delta_time: true
            """
        )
        state_space = (
            lambda x: 0.75 * np.sin(6.5 * x)
            + 1.15 * np.sin(3 * x)
            - 2.5 * np.sin(1.5 * x)
        )
        state_space_label = (
            r"State space: $y=0.75\;\sin(6.5\;x) + 1.15\;\sin(3 x) - 2.5\;\sin("
            r"1.5\;x)$"
        )
        omegaconf.OmegaConf.set_readonly(cfg, True)
        return cfg, state_space, state_space_label

    @pytest.fixture(scope="function")
    def setup_3D_env_cfg(
        self, setup_1D_env_cfg
    ) -> Tuple[omegaconf.DictConfig, Callable, str]:
        cfg, *_ = setup_1D_env_cfg
        with omegaconf.read_write(cfg):
            omegaconf.OmegaConf.update(
                cfg,
                "time_space.bound",
                [0, 200],
                # [0, 80],
                merge=False,
            )
            omegaconf.OmegaConf.update(
                cfg,
                "granularity",
                100000,
                # 10000,
                merge=False,
            )
        state_space = partial(
            rollout_lorenz_attractor_partial_derivative,
            **{"s": 10, "r": 28, "b": 2.667},
        )
        state_space_label = "Lorenz attractor"
        return cfg, state_space, state_space_label

    @staticmethod
    def make_math_env(cfg, math_fct, math_fct_label):
        env: Union[MathContinuousGymnasium, gym.Env] = gym.make(
            "math_gymnasium:math-continuous-gymnasium-v0",
            math_function_callback=math_fct,
            math_function_label=math_fct_label,
            time_axis_cfg=cfg.time_space,
            explorable_regions_cfg=cfg.explorable_space,
            measurement_noise_cfg=(
                cfg.global_measurement_noise,
                *cfg.measurement_noise,
            ),
            time_function_callback=None,
            observed_state_are_dt_derivatives=cfg.obs_are_dt_derivatives,
            observed_time_is_delta_time=cfg.obs_time_is_delta_time,
        )
        return env

    def test_init_general(self, setup_1D_env_cfg):
        cfg, math_fct, math_fct_label = setup_1D_env_cfg
        env = self.make_math_env(cfg, math_fct, math_fct_label)
        assert env.spec.kwargs["math_function_label"] == math_fct_label

        assert isinstance(env.get_wrapper_attr("_explorable_space_list"), list)
        assert len(env.get_wrapper_attr("_explorable_space_list")) == len(
            cfg.explorable_space
        )
        env.close()

    def test_trajectory_properties(self, setup_1D_env_cfg):
        cfg, math_fct, math_fct_label = setup_1D_env_cfg
        env = self.make_math_env(cfg, math_fct, math_fct_label)

        assert isinstance(env.trajectory, MathEnvTrajectoryDataclass)
        assert isinstance(env.trj, MathEnvTrajectoryDataclass)
        assert id(env.trajectory) != id(
            env.unwrapped._trajectory
        ), "`env.trajectory` should return a copy of `env._trajectory`"
        assert id(env.trj) != id(
            env.unwrapped._trajectory
        ), "env.trajectory should return a copy of `env._trajectory`"

        env.close()

    def test_init_1D(self, setup_1D_env_cfg):
        cfg, math_fct, math_fct_label = setup_1D_env_cfg
        env = self.make_math_env(cfg, math_fct, math_fct_label)
        assert env.observation_space.shape == (1,)
        assert env.trj.trajectory_len == cfg.time_space.granularity
        assert env.trj.state_axes.poses.shape == (cfg.time_space.granularity, 1)
        env.close()

    def test_init_3D(self, setup_3D_env_cfg):
        cfg, math_fct, math_fct_label = setup_3D_env_cfg
        env = self.make_math_env(cfg, math_fct, math_fct_label)
        assert env.observation_space.shape == (3,)
        assert env.trj.trajectory_len == cfg.time_space.granularity
        assert env.trj.state_axes.poses.shape == (cfg.time_space.granularity, 3)
        env.close()

    @pytest.mark.parametrize(
        argnames="t_bad_cfg_key,t_bad_cfg_value",
        argvalues=[
            ("time_space.bound", 0),
            ("time_space.bound", [0.0]),
            ("global_measurement_noise.interval", 0),
            ("global_measurement_noise.interval", [0.0]),
            ("global_measurement_noise.interval", [1.0, 0.0]),
            ("explorable_space", 0),
            ("explorable_space", [0.0]),
            ("explorable_space", [1.0, 0.0]),
        ],
    )
    def test_init_1D_bad_cfg(self, setup_1D_env_cfg, t_bad_cfg_key, t_bad_cfg_value):
        cfg, math_fct, math_fct_label = setup_1D_env_cfg

        with omegaconf.read_write(cfg):
            omegaconf.OmegaConf.update(cfg, t_bad_cfg_key, t_bad_cfg_value, merge=False)

        with pytest.raises(AssertionError) as exc_info:
            env = self.make_math_env(cfg, math_fct, math_fct_label)
            env.close()
        print(f"{exc_info=}")

    @pytest.mark.parametrize(
        argnames="t_bad_cfg_key,t_bad_cfg_value",
        argvalues=[
            ("time_space.bound", 0),
            ("time_space.bound", [0.0]),
            ("global_measurement_noise.interval", 0),
            ("global_measurement_noise.interval", [0.0]),
            ("global_measurement_noise.interval", [1.0, 0.0]),
            ("explorable_space", 0),
            ("explorable_space", [0.0]),
            ("explorable_space", [1.0, 0.0]),
        ],
    )
    def test_init_3D_bad_cfg(self, setup_3D_env_cfg, t_bad_cfg_key, t_bad_cfg_value):
        cfg, math_fct, math_fct_label = setup_3D_env_cfg

        with omegaconf.read_write(cfg):
            omegaconf.OmegaConf.update(cfg, t_bad_cfg_key, t_bad_cfg_value, merge=False)

        with pytest.raises(AssertionError) as exc_info:
            env = self.make_math_env(cfg, math_fct, math_fct_label)
            env.close()
        print(f"{exc_info=}")

    def test_reset_1D(self, setup_1D_env_cfg):
        cfg, math_fct, math_fct_label = setup_1D_env_cfg
        env = self.make_math_env(cfg, math_fct, math_fct_label)
        obs, info = env.reset(time_space_init_idx=0, explorable_space_only=False)
        assert info["curent_time_idx"] == 0
        assert isinstance(obs, dict)
        assert isinstance(obs["state_axes_obs_with_noise"], np.ndarray)
        assert isinstance(obs["time_axis_obs_with_noise"], float)
        assert obs["state_axes_obs_with_noise"].shape == (1,)
        env.close()

    def test_reset_3D(self, setup_3D_env_cfg):
        cfg, math_fct, math_fct_label = setup_3D_env_cfg

        env = self.make_math_env(cfg, math_fct, math_fct_label)
        obs, info = env.reset(time_space_init_idx=0, explorable_space_only=False)
        assert info["curent_time_idx"] == 0
        assert isinstance(obs, dict)
        assert isinstance(obs["state_axes_obs_with_noise"], np.ndarray)
        assert isinstance(obs["time_axis_obs_with_noise"], float)
        assert obs["state_axes_obs_with_noise"].shape == (3,)
        env.close()

    def test_reset_time_idx_out_of_explorable_bounds(self, setup_1D_env_cfg):
        cfg, math_fct, math_fct_label = setup_1D_env_cfg
        env = self.make_math_env(cfg, math_fct, math_fct_label)
        with pytest.raises(AssertionError):
            obs, info = env.reset(time_space_init_idx=0, explorable_space_only=True)
        env.close()

    def test_reset_time_idx_out_of_bounds(self, setup_1D_env_cfg):
        cfg, math_fct, math_fct_label = setup_1D_env_cfg
        env = self.make_math_env(cfg, math_fct, math_fct_label)
        with pytest.raises(AssertionError) as exc_info:
            obs, info = env.reset(
                time_space_init_idx=cfg.time_space.granularity,
                explorable_space_only=False,
            )
        print(f"{exc_info=}")
        assert exc_info.value.args == (
            "Param time_space_init_idx with arg 10000 out of bound",
        )
        env.close()

    def test_step_1D(self, setup_1D_env_cfg):
        cfg, math_fct, math_fct_label = setup_1D_env_cfg
        env = self.make_math_env(cfg, math_fct, math_fct_label)
        obs, info = env.reset(time_space_init_idx=0, explorable_space_only=False)
        next_obs, _, terminated, truncated, info = env.step()
        assert info["curent_time_idx"] == 1
        assert isinstance(obs, dict)
        assert isinstance(obs["state_axes_obs_with_noise"], np.ndarray)
        assert isinstance(obs["time_axis_obs_with_noise"], float)
        assert obs["state_axes_obs_with_noise"].shape == (1,)
        assert (terminated and truncated) is False
        env.close()

    def test_step_3D(self, setup_3D_env_cfg):
        cfg, math_fct, math_fct_label = setup_3D_env_cfg

        env = self.make_math_env(cfg, math_fct, math_fct_label)
        obs, info = env.reset(time_space_init_idx=0, explorable_space_only=False)
        next_obs, _, terminated, truncated, info = env.step()
        assert info["curent_time_idx"] == 1
        assert isinstance(obs, dict)
        assert isinstance(obs["state_axes_obs_with_noise"], np.ndarray)
        assert isinstance(obs["time_axis_obs_with_noise"], float)
        assert obs["state_axes_obs_with_noise"].shape == (3,)
        assert (terminated and truncated) is False
        env.close()

    def test_step_reach_end_of_time_space(self, setup_1D_env_cfg):
        cfg, math_fct, math_fct_label = setup_1D_env_cfg
        env = self.make_math_env(cfg, math_fct, math_fct_label)
        rollout_len = 10
        obs, _ = env.reset(
            time_space_init_idx=cfg.time_space.granularity - rollout_len,
            explorable_space_only=False,
        )
        terminated = False
        while not terminated:
            next_obs, _, terminated, truncated, info = env.step()

        assert terminated is True
        env.close()

    def test_step_reach_end_of_explorable_bounds(self, setup_1D_env_cfg):
        cfg, math_fct, math_fct_label = setup_1D_env_cfg
        env = self.make_math_env(cfg, math_fct, math_fct_label)
        rollout_len = 10
        explorable_right_bound = cfg.explorable_space[1][1]
        obs, _ = env.reset(
            time_space_init_idx=explorable_right_bound - rollout_len,
            explorable_space_only=True,
        )
        terminated = False
        while not terminated:
            next_obs, _, terminated, truncated, info = env.step()

        assert terminated is True
        env.close()

    def test_step_out_of_explorable_bounds(self, setup_1D_env_cfg):
        cfg, math_fct, math_fct_label = setup_1D_env_cfg
        env = self.make_math_env(cfg, math_fct, math_fct_label)
        rollout_len = 10
        explorable_right_bound = cfg.explorable_space[1][1]
        obs, _ = env.reset(
            time_space_init_idx=explorable_right_bound - rollout_len,
            explorable_space_only=True,
        )
        with pytest.raises(AssertionError) as exc_info:
            for _ in range(rollout_len + 1):
                next_obs, _, terminated, truncated, info = env.step()

        print(f"{exc_info=}")
        assert exc_info.value.args == (
            f"Time space index out of explorable space. Should have reset the environment "
            f"at previous step on 'terminated==True'",
        )
        env.close()

    @pytest.mark.parametrize(
        argnames="t_reset",
        argvalues=[False, True],
        ids=[
            "Execute _generate_new_obs_measurement_noise() at `__init__`",
            "Execute _generate_new_obs_measurement_noise() at `reset()`",
        ],
    )
    @pytest.mark.parametrize(
        argnames="t_dim", argvalues=[1, 3], ids=["Test 1dim case", "Test 3dim case"]
    )
    def test__generate_new_measurement_noise(
        self, setup_1D_env_cfg, setup_3D_env_cfg, t_reset, t_dim
    ):
        if t_dim == 1:
            cfg, math_fct, math_fct_label = setup_1D_env_cfg
        elif t_dim == 3:
            cfg, math_fct, math_fct_label = setup_3D_env_cfg

        env = self.make_math_env(cfg, math_fct, math_fct_label)

        if t_reset:
            obs, info = env.reset(explorable_space_only=False, seed=1234)

        explored = np.full_like(env.trj.state_axes.noise, fill_value=False)

        for each_measurement_noise_spec in [
            *env.get_wrapper_attr("_measurement_noise_cfg")
        ]:
            measurement_noise_interval = slice(*each_measurement_noise_spec["interval"])
            assert not np.array_equal(
                env.trj.state_axes.noise[measurement_noise_interval],
                np.zeros_like(env.trj.state_axes.noise[measurement_noise_interval]),
            )
            explored[measurement_noise_interval] = True

        assert np.array_equal(
            explored[explored == False],
            np.zeros_like(env.trj.state_axes.noise[explored == False]),
        )

        env.close()

    @pytest.mark.parametrize(
        argnames="t_dim", argvalues=[1, 3], ids=["Test 1dim case", "Test 3dim case"]
    )
    @pytest.mark.parametrize(
        argnames="t_obs_dt_derivatives",
        argvalues=[True, False],
        ids=["obs_are_dt_derivatives=True", "obs_are_dt_derivatives=False"],
    )
    def test__generate_new_measurement_noise_case_obs_derivative(
        self, setup_1D_env_cfg, setup_3D_env_cfg, t_dim, t_obs_dt_derivatives
    ):
        if t_dim == 1:
            cfg, math_fct, math_fct_label = setup_1D_env_cfg
        elif t_dim == 3:
            cfg, math_fct, math_fct_label = setup_3D_env_cfg

        with omegaconf.read_write(cfg):
            omegaconf.OmegaConf.update(
                cfg, "obs_are_dt_derivatives", t_obs_dt_derivatives, merge=False
            )

        env = self.make_math_env(cfg, math_fct, math_fct_label)

        assert np.isfinite(env.trj.state_axes.poses).any()
        assert np.isfinite(env.trj.state_axes.vels).any()
        assert np.isfinite(env.trj.state_axes.noise).any()
        assert np.isfinite(env.trj.state_axes.poses_with_noise).any()
        assert np.isfinite(env.trj.state_axes.obs_with_noise).any()

        assert np.isfinite(env.trj.time_axis.wall).any()
        assert np.isfinite(env.trj.time_axis.delta).any()
        assert np.isfinite(env.trj.time_axis.noise).any()
        assert np.isfinite(env.trj.time_axis.wall_with_noise).any()
        assert np.isfinite(env.trj.time_axis.obs_with_noise).any()

        obs_with_noise_ = env.trj.state_axes.obs_with_noise
        initiale_coordinates = env.trj.state_axes.poses_with_noise[0]
        if t_obs_dt_derivatives:
            obs_with_noise_ = convert_dt_state_derivatives_to_state_coordinate(
                obs_with_noise_, initiale_coordinates
            )

        # TRJ_LEN = slice(0, 100)
        TRJ_LEN = slice(0, env.trj.trajectory_len)
        NP_PRINT_PRECISION = 10
        numpy_array_print_precision_warning(NP_PRINT_PRECISION)
        with np.printoptions(
            precision=NP_PRINT_PRECISION,
            suppress=True,
            linewidth=100,
        ):
            print(
                f"poses_with_noise\n", env.trj.state_axes.poses_with_noise[TRJ_LEN, ...]
            )
            print(f"\nobs_with_noise_\n", obs_with_noise_[TRJ_LEN, ...])

        assert env.trj.state_axes.poses_with_noise[TRJ_LEN, ...] == pytest.approx(
            obs_with_noise_[TRJ_LEN, ...]
        )
        env.close()
