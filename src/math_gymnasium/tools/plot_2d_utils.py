# coding=utf-8
from typing import Tuple

import numpy as np
import omegaconf
import torch

from math_gymnasium.tools.style import *
from matplotlib import pyplot as plt


def prep_two_dimension_observation_prediction_output_for_plotting(
    y_pred: torch.Tensor, y_pred_logvar: torch.Tensor, obs_shape: tuple, ensemble_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares prediction output tensors for plotting by adjusting their dimensions as needed.
    Take tensor of the form `E x B x Out_Dim` or `B x Out_Dim` and return one of form `E x B`.

    :param y_pred: Tensor containing the predicted values.
    :param y_pred_logvar: Tensor containing the predicted log variance.
    :param obs_shape: The environment observations space shape.
    :param ensemble_size: Integer representing the size of the ensemble.
    :return: A tuple containing the adjusted predicted values tensor and log variance tensor.
    """
    if obs_shape[0] == 1:
        # Unpack output dimension: E x B x Out_Dim → E x B
        y_pred = y_pred.squeeze()
        y_pred_logvar = y_pred_logvar.squeeze()
    if ensemble_size == 1:
        # Add the missing ensemble dimension: B → E x B
        y_pred = torch.unsqueeze(y_pred, 0)
        y_pred_logvar = torch.unsqueeze(y_pred_logvar, 0)
    return y_pred, y_pred_logvar


def two_dimension_environment_space_plot(
    cfg: omegaconf.DictConfig,
    state_space_x: np.ndarray,
    state_space_y: np.ndarray,
    state_space_y_with_noise: np.ndarray,
    title: str,
    state_space_label: str,
    show_samples: bool = True,
    figsize: tuple = (20, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=50) # default dpi=100

    ax.plot(state_space_x,
            state_space_y,
            color=COLOR_GROUND_TRUTH,
            alpha=COLOR_GROUND_TRUTH_ALPHA,
            ls="--",
            linewidth=2,
            label=state_space_label)

    # ax.plot(state_space_x, target_state_space_y, '.', color='r', markersize=1.9, alpha=0.25,
    # label=state_space_label_)
    show_explorable_label_once = "Explorable region of the state space"
    show_sample_label_once = "Sample's"
    for each_explorable_interval_ in [*cfg.environment.explorable_space]:
        interval_ = slice(each_explorable_interval_[0], each_explorable_interval_[1])
        if show_samples:
            ax.plot(
                state_space_x[interval_],
                state_space_y_with_noise[interval_],
                ".",
                color=COLOR_OBSERVATIONS,
                markersize=MARKERSIZE_OBSERVATIONS,
                alpha=COLOR_OBSERVATIONS_ALPHA,
                label=show_sample_label_once,
            )
        ax.axvspan(
            state_space_x[interval_.start],
            state_space_x[interval_.stop - 1],
            linestyle="-",
            linewidth=1,
            color="gray",
            alpha=0.12,
            label=show_explorable_label_once,
        )
        show_explorable_label_once = ""
        show_sample_label_once = ""
    ax.set_ylabel(r"$y$", fontsize=20)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1, -0.1))
    ax.grid(True)
    ax.axis([*cfg.environment.time_space.bound, -6, 6])
    ax.set_title(title, size="large", weight="bold")
    fig.tight_layout(pad=1)
    return fig, ax


def two_dimension_prediction_plot(
    cfg: omegaconf.DictConfig,
    state_space_x: np.ndarray,
    state_space_y_base: np.ndarray,
    state_space_y_target: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    title: str,
    state_space_label: str,
    figsize: tuple = (20, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = two_dimension_environment_space_plot(
        cfg,
        state_space_x,
        state_space_y_base,
        state_space_y_target,
        title,
        state_space_label,
        show_samples=False,
        figsize=figsize,
    )
    state_space_x = state_space_x[1:]
    state_space_y_target = state_space_y_target[1:]
    y_pred = y_pred[:-1]
    y_std = y_std[:-1]

    # .... Observations ...........................................................................
    ax.plot(
        state_space_x,
        state_space_y_target,
        ".",
        color=COLOR_OBSERVATIONS,
        markersize=MARKERSIZE_OBSERVATIONS,
        alpha=COLOR_OBSERVATIONS_ALPHA,
        label="Target env measurement noise",
    )

    # .... Predictions ............................................................................
    ax.plot(state_space_x, y_pred, ".",
            color=COLOR_PREDICTIONS,
            markersize=MARKERSIZE_PREDICTIONS,
            alpha=COLOR_PREDICTIONS_ALPHA,
            label="Prediction")

    # .... Aleatoric uncertainty standard deviation ...............................................
    ax.fill_between(
        state_space_x,
        y_pred,
        y_pred + 2 * y_std,
        color=COLOR_ALE,
        alpha=COLOR_ALE_ALPHA,
        label="Uncertainty (ale + epi)",
    )

    ax.fill_between(state_space_x, y_pred - 2 * y_std, y_pred,
                    color=COLOR_ALE,
                    alpha=COLOR_ALE_ALPHA
                    )

    # .... Epistemic uncertainty standard deviation ...............................................
    # (Priority) ToDo: implement EPI fill between.
    #  Ref _setup_1d_axis_prediction_subplot() fct at
    #  utilities/math-gymnasium/src/math_gymnasium/tools/plot_3d_utils.py:522

    plt.legend(loc="lower right", bbox_to_anchor=(1, -0.18))
    return fig, ax
