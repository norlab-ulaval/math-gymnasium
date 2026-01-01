# coding=utf-8
from typing import Union

import numpy as np

from math_gymnasium.envs.arbitrary_dim_math_continuous import (
    MathContinuousGymnasium,
)
from trajectory_container_tools.dataclasses.erll_trajectory_dataclass import (
    TestMotionTrajectoryDataclass,
)
from trajectory_container_tools.dataclasses.math_gymnasium_trajectory_dataclass import (
    MathEnvTrajectoryDataclass,
)


def math_continuous_gymnasium_env_to_test_motion_trajectory_dataclass(
    test_env: Union[MathEnvTrajectoryDataclass, MathContinuousGymnasium],
) -> TestMotionTrajectoryDataclass:

    trj: MathEnvTrajectoryDataclass
    if isinstance(test_env, MathEnvTrajectoryDataclass):
        trj = test_env
    elif isinstance(test_env.unwrapped, MathContinuousGymnasium):
        trj = test_env.trajectory
    else:
        raise NotImplementedError(
            "Param `test_env` curently only support type `MathEnvTrajectoryDataclass` and "
            f"`MathContinuousGymnasium` but `{type(test_env)}` was given."
        )

    return TestMotionTrajectoryDataclass(
        feature_name=f"{trj.feature_name} test trajectory",
        observations=trj.state_axes.obs_with_noise,
        actions=trj.time_axis.obs_with_noise,
        pose=trj.state_axes.poses_with_noise,
        pose_gt=trj.state_axes.poses,
    )


def agreement_score_relative_exp(
    x: np.ndarray, scale: float = 1.0, eps: float = 1e-12
) -> float:
    """
    Computes an agreement score based on a relative exponential model.

    :param x: An array of numeric values for which the agreement score is computed.
    :param scale: A positive scaling factor that affects the agreement computation.
        Smaller => harsher penalty.
    :param eps: A small positive number added to prevent division by zero.
    :return: A scale-invariant agreement score in (0,1].
    :raises ValueError: If the provided scale is not greater than zero.
    """
    x = np.asarray(x, dtype=float)
    denom = abs(float(x.mean())) + eps
    r = float((x.max() - x.min()) / denom)
    if scale <= 0:
        raise ValueError("scale must be > 0")
    return float(np.exp(-r / scale))
