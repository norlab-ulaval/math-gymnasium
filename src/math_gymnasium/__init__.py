# coding=utf-8
"""
Math Gymnasim (MG)

Quick Start:

>>> import math_gymnasium as mg

"""

from . import envs
from . import tools
from . import dynamical_systems

from gymnasium.envs.registration import register
from gymnasium.envs.registration import WrapperSpec

register(
    id="math-continuous-gymnasium-v0",
    entry_point=(
        "math_gymnasium.envs.arbitrary_dim_math_continuous:MathContinuousGymnasium"
    ),
    additional_wrappers=(
        WrapperSpec(
            name="AttributeFetchingWrapper",
            entry_point="math_gymnasium.envs.math_continuous_wrapper:AttributeFetchingWrapper",
            kwargs={},  # Note: Use empty dict instead of None
        ),
    ),
)


__all__ = [
        "tools",
        "envs",
        "dynamical_systems",
    ]
