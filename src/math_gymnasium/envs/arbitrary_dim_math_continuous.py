# coding=utf-8
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, SupportsFloat, Tuple, Union

import omegaconf
import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, RenderFrame

from tools.math_tools.space_conversion_tools.coordinate_to_velocity import (
    convert_state_coordinate_to_state_derivatives,
)
from tools.math_tools.space_conversion_tools.time_to_delta_time import (
    convert_state_time_to_state_delta_time,
)
from tools.math_tools.space_conversion_tools.utils import (
    convert_ndim_data_to_delta_ndim_data,
)
from tools.math_tools.ndarray_tools.custom_msg import nan_infinity_console_warning

from trajectory_container_tools.dataclasses.math_gymnasium_trajectory_dataclass import (
    MathEnvTrajectoryDataclass,
    StateAxDataclass,
    TimeAxDataclass,
)


class MathContinuousGymnasium(gym.Env):
    _explorable_space_list: List[Tuple[int, int]]
    _measurement_noise_cfg: List[Dict[str, Union[Tuple[int, int], float]]]
    _trajectory: MathEnvTrajectoryDataclass

    metadata = {"render_modes": []}

    def __init__(
        self,
        math_function_callback: Callable[[np.ndarray], np.ndarray],
        math_function_label: str,
        time_axis_cfg: omegaconf.DictConfig,
        explorable_regions_cfg: omegaconf.ListConfig,
        measurement_noise_cfg: Optional[Tuple[omegaconf.DictConfig]] = None,
        time_function_callback: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        observed_state_are_dt_derivatives: bool = True,
        observed_time_is_delta_time: bool = True,
    ):
        """
        Mathematic function formulated as a gymnasium environment. This is a no action
        environment meant for evaluating model-based RL motion dynamic model learning quality.
        Mathematical function that can be visualize in 2D or 3D such as periodic time series and
        chaotic time series are a good tools for that purpose.

        Think of a no action environment as any other gymnasium environment for which we would
        purposily freeze the action space to a unique trajectory rollout instance. The only
        thing that can then vary is the observation noise and the environment hyperparameter
        configuration.

        From an RL point of view, picture that the environment assign the math function output
        to the observation space and the math function input to the action space.
         Example:
            >>> action space: x
            >>> observation space: y = sin(x)

        Note:
        =====

        - New measurement noise is generated at initialisation and when executing `reset`.
        - Param `measurement_noise_cfg` argument must take the form:

            >>> [
            >>>   {
            >>>     'interval': [<start-index>, <end-index>]
            >>>     'magnitude': <noise-standard-deviation>
            >>>   },
            >>>   ...
            >>> ]

        - Param `time_axis_cfg` argument must take the form:

            >>> {
            >>>   'bound': [<start-index>, <end-index>]
            >>>   'granularity': <time-space-array-size>
            >>> }

        - Param `explorable_regions_cfg` argument must take the form:

            >>> [(start_index, end_index),...]



        :param math_function_callback: A callable that defines the state-space model.
        :param math_function_label: A string representation of the mathematical use function for
        ploting.
        :param time_axis_cfg: A time space config dict, containing `granularity` and `bounds`.
        :param explorable_regions_cfg: A list of tuple defining the explorable
         region in the input space i.e `[(start_index, end_index),...]`.
        :param measurement_noise_cfg: An optional list of configuration dictionaries
         specifying `intervals` and `magnitudes` for measurement noise.
        :param time_function_callback: A callable that will be executed before feeding
         the `time_space` array to the `math_function_callback`.
        :param observed_state_are_dt_derivatives: True will set `reset` and `step` method to
         output `coordinate-derivative X delta-time`. Output `coordinate` otherwise.
        :param observed_time_is_delta_time: True will set observation to output
         `delta-time`. Output `time` otherwise.
        """

        self._observed_state_are_dt_derivatives = observed_state_are_dt_derivatives
        self._observed_time_is_delta_time = observed_time_is_delta_time
        self._curent_time_idx = None
        self._explorable_space_only = None

        # .... Pre-condition ......................................................................
        # (NICE TO HAVE) ToDo: implement check to validate math_function_callback signature
        # (kwarg and return).
        assert isinstance(math_function_callback, Callable)
        assert isinstance(math_function_label, str)
        if time_function_callback is not None:
            assert isinstance(time_function_callback, Callable)

        assert omegaconf.OmegaConf.is_config(time_axis_cfg)
        assert time_axis_cfg.granularity, f"missing cfg key 'granularity'"
        assert time_axis_cfg.bound, f"missing cfg key 'bound'"
        _check_interval_cfg(time_axis_cfg.bound, "time_space.bound")

        if measurement_noise_cfg:
            if isinstance(measurement_noise_cfg, omegaconf.DictConfig):
                measurement_noise_cfg = (measurement_noise_cfg,)
            each_noise_spec: omegaconf.DictConfig
            for each_noise_spec in [*measurement_noise_cfg]:
                assert omegaconf.OmegaConf.is_config(
                    each_noise_spec
                ), f"`measurement_noise` list does not contain omegaconf cfg dictionary"
                assert (
                    omegaconf.OmegaConf.select(each_noise_spec, "magnitude") is not None
                ), f"missing cfg key 'magnitude'"
                assert (
                    omegaconf.OmegaConf.select(each_noise_spec, "interval") is not None
                ), f"missing cfg key 'interval'"
                _check_interval_cfg(
                    each_noise_spec.interval, "each_noise_spec.interval"
                )

        assert omegaconf.OmegaConf.is_list(
            explorable_regions_cfg
        ), f"state_space_explorable_region is not a list"
        for each_explorable_region in [*explorable_regions_cfg]:
            _check_interval_cfg(each_explorable_region, "state_space_explorable_region")

        # .... Freeze input configuration .........................................................
        omegaconf.OmegaConf.set_readonly(time_axis_cfg, True)
        omegaconf.OmegaConf.set_readonly(explorable_regions_cfg, True)
        for each_cfg in measurement_noise_cfg:
            omegaconf.OmegaConf.set_readonly(each_cfg, True)

        # .... Setup internal time space ..........................................................
        time_axis = np.linspace(
            *time_axis_cfg.bound, time_axis_cfg.granularity, endpoint=False
        )
        if isinstance(time_function_callback, Callable):
            time_axis = time_function_callback(time_axis)

        # .... Setup internal state spaces ........................................................
        if isinstance(math_function_callback, Callable):
            state_axes = math_function_callback(time_axis)
            if state_axes.ndim == 1:
                state_axes = np.expand_dims(state_axes, 1)
            dtype = state_axes.dtype
        else:
            raise NotImplementedError(
                "math_function must be either a function that return a numpy ndarray or a "
                "R2SMotionModelContainer with a pre-trained model"
            )
        assert isinstance(
            state_axes, np.ndarray
        ), f"math_function must return a np.ndarray"
        time_axis = time_axis.astype(dtype)
        state_axes_derivatives = convert_state_coordinate_to_state_derivatives(
            state_axes, time_axis, False
        )
        delta_time_axis = convert_state_time_to_state_delta_time(time_axis)

        # .... Setup internal observed time/state spaces ..........................................
        if self._observed_state_are_dt_derivatives:
            state_axes_obs = state_axes_derivatives * np.expand_dims(delta_time_axis, 1)
        else:
            state_axes_obs = state_axes

        if self._observed_time_is_delta_time:
            time_axis_obs = delta_time_axis
        else:
            time_axis_obs = time_axis

        # .... Register trajectory components and validate data ...................................
        self._trajectory = MathEnvTrajectoryDataclass(
            feature_name=math_function_label,
            state_axes=StateAxDataclass(
                poses=state_axes,
                vels=state_axes_derivatives,
                obs=state_axes_obs,
            ),
            time_axis=TimeAxDataclass(
                wall=time_axis,
                delta=delta_time_axis,
                obs=time_axis_obs,
            ),
        )

        self._validate_and_register_measurement_noise(measurement_noise_cfg)
        self._validate_and_register_explorable_space(explorable_regions_cfg)
        self._generate_new_obs_measurement_noise()

        # .... Setup gym state/action spaces ......................................................
        self.action_space = gym.spaces.Box(
            low=time_axis_cfg.bound[0],
            high=time_axis_cfg.bound[1],
            shape=(1,),
            dtype=dtype,
        )

        if state_axes.ndim == 1:
            high = np.inf
            obs_space_shape = (1,)
        else:
            obs_space_shape = (state_axes.shape[-1],)
            high = np.full(obs_space_shape, np.inf)

        self.observation_space = gym.spaces.Box(
            low=-high,
            high=high,
            shape=obs_space_shape,
            dtype=dtype,
        )

    @property
    def trajectory(self) -> MathEnvTrajectoryDataclass:
        return deepcopy(self._trajectory)

    @property
    def trj(self) -> MathEnvTrajectoryDataclass:
        """Shortcut for `trajectory`"""
        return deepcopy(self._trajectory)

    def _validate_and_register_measurement_noise(
        self, measurement_noise_cfg: Optional[Tuple[omegaconf.DictConfig]]
    ) -> None:
        """Validates and register measurement noise spec."""
        if measurement_noise_cfg[0] is not None:
            self._measurement_noise_cfg = []
            for idx, each_noise_spec in enumerate([*measurement_noise_cfg]):
                self._measurement_noise_cfg.append(
                    omegaconf.OmegaConf.to_object(each_noise_spec)
                )
                measurement_noise_interval = slice(*each_noise_spec.interval)
                assert (
                    measurement_noise_interval.stop <= self._trajectory.trajectory_len
                ), f"{measurement_noise_interval.stop=} !< {self._trajectory.trajectory_len=}"
                assert each_noise_spec.magnitude >= 0.0
        else:
            self._measurement_noise_cfg = None

        return None

    def _validate_and_register_explorable_space(
        self, state_space_explorable_region: omegaconf.ListConfig
    ) -> None:
        """Register explorable region of the time space and validates each interval's bounds."""
        self._explorable_space_list = omegaconf.OmegaConf.to_object(
            state_space_explorable_region
        )
        for each_explorable_interval in self._explorable_space_list:
            # each_explorable_interval[1] -= 1
            explorable_interval_slice = slice(*each_explorable_interval)
            assert (
                explorable_interval_slice.start < explorable_interval_slice.stop
            ), f"{explorable_interval_slice.start=} !< {explorable_interval_slice.stop=}"
            assert (
                explorable_interval_slice.stop <= self._trajectory.trajectory_len
            ), f"{explorable_interval_slice.stop=} !<= {self._trajectory.trajectory_len=}"

        return None

    def _generate_new_obs_measurement_noise(self) -> None:
        """Sets and validates the state space with measurement noise."""
        self._trajectory.state_axes.noise = np.zeros_like(
            self._trajectory.state_axes.noise
        )
        if self._measurement_noise_cfg is not None:
            for each_measurement_noise_spec in [*self._measurement_noise_cfg]:
                measurement_noise_interval = slice(
                    *each_measurement_noise_spec["interval"]
                )
                assert (
                    measurement_noise_interval.start < measurement_noise_interval.stop
                )
                zero_mean_noise = self.np_random.standard_normal(
                    size=self._trajectory.state_axes.noise[
                        measurement_noise_interval
                    ].shape,
                    dtype=np.float64,
                )
                self._trajectory.state_axes.noise[measurement_noise_interval] = (
                    zero_mean_noise * each_measurement_noise_spec["magnitude"]
                ).astype(self._trajectory.state_axes.noise.dtype)
            if self._observed_state_are_dt_derivatives:
                # Note: Using noise-delta instead of noise is the way to go when observation
                # are coordinate derivatives.
                self._trajectory.state_axes.obs_noise = (
                    convert_ndim_data_to_delta_ndim_data(
                        self._trajectory.state_axes.noise
                    )
                )
            else:
                self._trajectory.state_axes.obs_noise = (
                    self._trajectory.state_axes.noise
                )

        if not np.all(np.isfinite(self._trajectory.state_axes.obs_with_noise)):
            nan_infinity_console_warning(
                "MathContinuousGymnasium._trajectory.state_axes.obs_with_noise"
            )

        return None

    def reset(
        self,
        *,
        time_space_init_idx: Union[int, np.int64] = 0,
        explorable_space_only: bool = True,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Resets the environment to an initial state.

        :param time_space_init_idx: The initial time space index for resetting the environment.
        :param explorable_space_only: Restrict the rollout to explorable space (raise an error).
        :param seed: The random seed for reproducibility.
        :param options: Additional options for the reset process.
        :return: A tuple containing the observation and additional info dictionary.
        """
        super().reset(seed=seed, options=options)

        assert isinstance(
            time_space_init_idx, (int, np.int64)
        ), f"time_space_init_idx must be an int, currently {type(time_space_init_idx)}"
        assert (
            time_space_init_idx < self._trajectory.trajectory_len
        ), f"Param time_space_init_idx with arg {time_space_init_idx} out of bound"

        self._curent_time_idx = time_space_init_idx
        self._explorable_space_only = explorable_space_only

        if self._explorable_space_only:
            is_in_explorable_region = False
            for each_explorable_interval in self._explorable_space_list:
                is_in_explorable_region = (
                    each_explorable_interval[0]
                    <= self._curent_time_idx
                    <= each_explorable_interval[1]
                ) or is_in_explorable_region
            assert is_in_explorable_region, f"Time space index out of explorable space"

        self._generate_new_obs_measurement_noise()
        observation = {
            "state_axes_obs_with_noise": self._trajectory.state_axes.obs_with_noise[
                self._curent_time_idx
            ],
            "time_axis_obs_with_noise": self._trajectory.time_axis.obs_with_noise[
                self._curent_time_idx
            ],
        }
        info = {"curent_time_idx": self._curent_time_idx}
        return observation, info

    def step(
        self, action: ActType = None
    ) -> Tuple[Dict[str, np.ndarray], SupportsFloat, bool, bool, Dict[str, Any]]:
        assert action is None, (
            f"action are implicitly set by the environment and correspond to "
            f"incrementing the time space index by one. Set param 'action=None'"
        )
        self._curent_time_idx += 1
        terminated = False
        if self._explorable_space_only:
            is_in_explorable_region = False
            for each_explorable_interval in self._explorable_space_list:
                if self._curent_time_idx == each_explorable_interval[1]:
                    terminated = True

                is_in_explorable_region = (
                    each_explorable_interval[0]
                    <= self._curent_time_idx
                    <= each_explorable_interval[1]
                ) or is_in_explorable_region

            assert is_in_explorable_region, (
                f"Time space index out of explorable space. Should have reset the environment "
                f"at previous step on 'terminated==True'"
            )
        if self._curent_time_idx + 1 == self._trajectory.trajectory_len:
            terminated = True

        next_observation = {
            "state_axes_obs_with_noise": self._trajectory.state_axes.obs_with_noise[
                self._curent_time_idx
            ],
            "time_axis_obs_with_noise": self._trajectory.time_axis.obs_with_noise[
                self._curent_time_idx
            ],
        }
        reward = 1.0
        terminated = terminated
        truncated = False
        info = {"curent_time_idx": self._curent_time_idx}
        return next_observation, reward, terminated, truncated, info

    def render(self) -> Union[RenderFrame, List[RenderFrame]]:
        pass


def _check_interval_cfg(
    interval_cfg: Union[list, omegaconf.ListConfig], interval_cfg_key_str: str
) -> None:
    assert isinstance(
        interval_cfg, (list, omegaconf.ListConfig)
    ), f"{interval_cfg_key_str} is not a list"
    assert isinstance(interval_cfg_key_str, str)
    assert len(interval_cfg) == 2, f"{interval_cfg_key_str} length != 2"
    assert interval_cfg[0] < interval_cfg[1], (
        f"{interval_cfg_key_str}[0]={interval_cfg[0]} !< {interval_cfg_key_str}[1]="
        f"{interval_cfg[1]}"
    )
    return None
