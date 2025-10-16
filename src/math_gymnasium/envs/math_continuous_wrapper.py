# coding=utf-8

from gymnasium import Wrapper
from math_gymnasium.envs.arbitrary_dim_math_continuous import MathContinuousGymnasium


class AttributeFetchingWrapper(Wrapper):
    """
    A convenience wrapper that provides enhanced access to attributes and methods
    of MathContinuousGymnasium environment. The goal is to maintain the ability to easily get
    attribute 'trj' and 'trajectory' as was the case in Gymnasium < v1.0.

    This class is designed to fetch specific attributes from the unwrapped environment, making them
    directly accessible through the wrapper class.

    Note: overriding methods "reset" and "step" is necessary since MathContinuousGymnasium.reset
    as a non standard signature and MathContinuousGymnasium.step can legaly take no action.

    :ivar env: The wrapped environment instance.
    :type env: MathContinuousGymnasium
    """
    def __init__(self, env: MathContinuousGymnasium):
        super().__init__(env)

    @property
    def trajectory(self):
        return self.unwrapped.trajectory

    @property
    def trj(self):
        return self.unwrapped.trj

    def reset(self, **kwargs):
        return self.env.unwrapped.reset(**kwargs)

    def step(self, action=None):
        return self.env.unwrapped.step(action)
