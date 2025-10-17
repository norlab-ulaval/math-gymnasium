# coding=utf-8
from typing import Union

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
    )
