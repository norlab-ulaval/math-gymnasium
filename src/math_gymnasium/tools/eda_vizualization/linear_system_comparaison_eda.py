import numpy as np
import matplotlib.pyplot as plt

from math_gymnasium.dynamical_systems import (
    rollout_damped_oscillator_partial_derivative,
    rollout_linear_spiral_partial_derivative,
)

# Compare the x-y plane dynamics (ignoring z)
time_space = np.linspace(0, 20, 2000)

# Linear Spiral
linear_traj = rollout_linear_spiral_partial_derivative(
    time_space,
    a=-0.1,
    omega=1.0,
    c=0.0,  # Set to 0 to focus on x-y dynamics
    initiale_coordinates=(1.0, 0.0, 0.0),
)

# Damped Oscillator
damped_traj = rollout_damped_oscillator_partial_derivative(
    time_space,
    gamma=0.1,
    omega0=1.0,
    vz=0.0,  # Set to 0 to focus on x-y dynamics
    initiale_coordinates=(1.0, 0.0, 0.0),
)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Linear Spiral - Phase Space
axes[0, 0].plot(linear_traj[:, 0], linear_traj[:, 1])
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y (independent variable)")
axes[0, 0].set_title("Linear Spiral: Phase Space\n(x and y are independent)")
axes[0, 0].grid(True)

# Linear Spiral - Time Series
axes[0, 1].plot(time_space, linear_traj[:, 0], label="x")
axes[0, 1].plot(time_space, linear_traj[:, 1], label="y")
axes[0, 1].set_xlabel("time")
axes[0, 1].set_ylabel("value")
axes[0, 1].set_title("Linear Spiral: Time Series")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Linear Spiral - Velocity
linear_velocity = np.diff(linear_traj[:, 0]) / np.diff(time_space)
axes[0, 2].plot(time_space[:-1], linear_velocity, label="dx/dt", alpha=0.7)
axes[0, 2].plot(time_space, linear_traj[:, 1], label="y", alpha=0.7)
axes[0, 2].set_xlabel("time")
axes[0, 2].set_ylabel("value")
axes[0, 2].set_title("Linear Spiral: dx/dt vs y\n(NOT equal)")
axes[0, 2].legend()
axes[0, 2].grid(True)

# Damped Oscillator - Phase Space
axes[1, 0].plot(damped_traj[:, 0], damped_traj[:, 1])
axes[1, 0].set_xlabel("x (position)")
axes[1, 0].set_ylabel("y (velocity = dx/dt)")
axes[1, 0].set_title("Damped Oscillator: Phase Space\n(y = dx/dt)")
axes[1, 0].grid(True)

# Damped Oscillator - Time Series
axes[1, 1].plot(time_space, damped_traj[:, 0], label="x (position)")
axes[1, 1].plot(time_space, damped_traj[:, 1], label="y (velocity)")
axes[1, 1].set_xlabel("time")
axes[1, 1].set_ylabel("value")
axes[1, 1].set_title("Damped Oscillator: Time Series")
axes[1, 1].legend()
axes[1, 1].grid(True)

# Damped Oscillator - Velocity Verification
damped_velocity = np.diff(damped_traj[:, 0]) / np.diff(time_space)
axes[1, 2].plot(time_space[:-1], damped_velocity, label="dx/dt (computed)", alpha=0.7)
axes[1, 2].plot(
    time_space, damped_traj[:, 1], label="y (from system)", alpha=0.7, linestyle="--"
)
axes[1, 2].set_xlabel("time")
axes[1, 2].set_ylabel("value")
axes[1, 2].set_title("Damped Oscillator: dx/dt vs y\n(EQUAL - y IS dx/dt)")
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()
plt.show()
