# coding=utf-8
from typing import Optional, Tuple

import numpy as np
import omegaconf

import torch
from matplotlib import pyplot as plt

from math_gymnasium.tools.style import *
from math_gymnasium.tools.utils import agreement_score_relative_exp
from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run
from tools.math_tools.math_fct import numpy_softplus
from tools.plot_tools.style import AXIS_LABEL_STYLE

DIM_X = 0
DIM_Y = 1
DIM_Z = 2
LEGEND_BBOX_TO_ANCHOR = (1.0, 0.99)


def prep_arbitrary_dimension_observation_prediction_output_for_1D_plotting(
    selected_dimension: int,
    pred: torch.Tensor,
    pred_logvar: torch.Tensor,
    ensemble_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares prediction output tensors for plotting by adjusting their dimensions as needed.
    Take tensor of the form `E x B x Out_Dim` or `B x Out_Dim` and return one of form `E x B`.

    :param pred: Tensor containing the predicted values.
    :param pred_logvar: Tensor containing the predicted log variance.
    :param selected_dimension: The environment observations dimension to show.
    :param ensemble_size: Integer representing the size of the ensemble.
    :return: A tuple containing the adjusted predicted values tensor and log variance tensor.
    """
    # Unpack output dimension: E x B x Out_Dim → E x B x 1
    pred_sdim = pred[..., selected_dimension]
    pred_logvar_sdim = pred_logvar[..., selected_dimension]

    # Unpack output dimension: E x B x 1 → E x B
    pred_sdim = pred_sdim.squeeze()
    pred_logvar_sdim = pred_logvar_sdim.squeeze()

    if ensemble_size == 1:
        # Add the missing ensemble dimension: B → E x B
        pred_sdim = torch.unsqueeze(pred_sdim, 0)
        pred_logvar_sdim = torch.unsqueeze(pred_logvar_sdim, 0)
    return pred_sdim, pred_logvar_sdim


def three_dimension_environment_space_plot(
    cfg: omegaconf.DictConfig,
    time_space: np.ndarray,
    state_space_3d: np.ndarray,
    state_space_3d_with_noise: np.ndarray,
    title: str,
    state_space_label: str,
    subplot_1d_interval: Optional[slice] = None,
    show_samples: bool = True,
    figsize: tuple = (20, 16),
    figdpi: int = 50,
    show_explorable_space=True,
    extra_info_str: str = None,
    experiment_id: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
    omegaconf.OmegaConf.set_readonly(cfg, True)

    if subplot_1d_interval is None:
        subplot_1d_interval = slice(0, time_space.size - 1)
    else:
        assert isinstance(subplot_1d_interval, slice)

    fig: plt.Figure = plt.figure(figsize=figsize, dpi=figdpi)  # default dpi=100

    gs = fig.add_gridspec(3, 8)

    # .... Setup 3D plot ..........................................................................
    # ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_3d = fig.add_subplot(gs[:, 0:3], projection="3d")

    # .... Theoretical environment ................................................................
    ax_3d.plot3D(
        *state_space_3d[subplot_1d_interval, :].T,
        # *state_space_3d_with_noise[subplot_1d_interval, :].T,
        color=COLOR_GROUND_TRUTH,
        alpha=COLOR_LINE_3D_ALPHA,
        # alpha=1.0,
        lw=THREE_DIM_LW,
        zorder=4,  # Put in front of predictions
    )

    # Quick-hack to prevent 3D aspect ratio skewing
    # Credit: https://stackoverflow.com/a/72928548
    limits = np.array([getattr(ax_3d, f"get_{axis}lim")() for axis in "xyz"])
    ax_3d.set_box_aspect(np.ptp(limits, axis=1))

    # .... Observations ...........................................................................
    lower_limits = limits[..., 0]
    upper_limits = limits[..., 1]
    axis_len = upper_limits - lower_limits
    limits_ratio = agreement_score_relative_exp(axis_len, scale=1.0)

    # Shrink noise visualization below a treshold
    noise_dynamic_size = (
        np.absolute(
            state_space_3d_with_noise[subplot_1d_interval, :]
            - state_space_3d[subplot_1d_interval, :]
        )
    ).mean(axis=-1)

    noise_dynamic_size = 2.5 * numpy_softplus(noise_dynamic_size, beta=0.75)
    noise_dynamic_size = noise_dynamic_size.clip(max=4.0) * 0.5

    # Visualize noise "blur/glow" layers
    # three_d_noise_blur_cfg = [(6, 0.025), (3, 0.08), (1.5, 0.5)]
    three_d_noise_blur_cfg = [(5.0, 0.05), (2.5, 0.175), (1.0, 1.0)]
    for each_z, (each_ms, each_a) in enumerate(three_d_noise_blur_cfg):
        ax_3d.scatter3D(
            *state_space_3d_with_noise[subplot_1d_interval, :].T,
            marker="o",
            s=THREE_DIM_LW
            * COLOR_GROUND_TRUTH_NOISE_BLUR_MARKERSIZE
            * noise_dynamic_size**2
            * each_ms**2 * limits_ratio,
            color=COLOR_GROUND_TRUTH_NOISE_BLUR[each_z],
            alpha=COLOR_GROUND_TRUTH_NOISE_BLUR_ALPHA * each_a,
            zorder=each_z + 1,  # Put in behind the system ground thruth
            depthshade=True,  # keeps your glow alpha/color more faithful
            linewidths=0.0,
        )

    # Pane color
    ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Axes colors
    ax_3d.xaxis.line.set_color("gray")
    ax_3d.yaxis.line.set_color("gray")
    ax_3d.zaxis.line.set_color("gray")

    # Tick colors
    ax_3d.tick_params(axis="x", colors="gray")
    ax_3d.tick_params(axis="y", colors="gray")
    ax_3d.tick_params(axis="z", colors="gray")

    ax_3d.set_xlabel("X Axis", **AXIS_LABEL_STYLE)
    ax_3d.set_ylabel("Y Axis", **AXIS_LABEL_STYLE)
    ax_3d.set_zlabel("Z Axis", **AXIS_LABEL_STYLE)
    # ax_3d.set_title(
    #         state_space_label,
    #         loc="center",
    #         y=0.9,
    #         # pad=10,
    #         size="large",
    #         # size="medium",
    #         # weight="bold",
    #         color="gray",
    #         )

    # .... Setup xyz axes 1D subplot ..............................................................
    # fig.add_subplot(nrows, ncols, num)

    # ax_z = fig.add_subplot(2, 2, 2)
    ax_z = fig.add_subplot(gs[0, 3:])
    ax_z = _setup_1d_axis_subplot(
        ax_z,
        cfg,
        show_samples,
        state_space_3d,
        state_space_3d_with_noise,
        time_space,
        subplot_1d_interval,
        selected_dimension=DIM_Z,
        label=state_space_label,
        skip_label=False,
        show_explorable_space=show_explorable_space,
    )

    # ax_y = fig.add_subplot(2, 2, 4)
    ax_y = fig.add_subplot(gs[1, 3:], sharex=ax_z)
    ax_y = _setup_1d_axis_subplot(
        ax_y,
        cfg,
        show_samples,
        state_space_3d,
        state_space_3d_with_noise,
        time_space,
        subplot_1d_interval,
        selected_dimension=DIM_Y,
        skip_label=True,
        show_explorable_space=show_explorable_space,
    )

    # ax_x = fig.add_subplot(2, 2, 3)
    ax_x = fig.add_subplot(gs[2, 3:], sharex=ax_z)
    ax_x = _setup_1d_axis_subplot(
        ax_x,
        cfg,
        show_samples,
        state_space_3d,
        state_space_3d_with_noise,
        time_space,
        subplot_1d_interval,
        selected_dimension=DIM_X,
        skip_label=True,
        show_explorable_space=show_explorable_space,
    )

    ax_z.set_ylabel("z", fontsize=16, **AXIS_LABEL_STYLE)
    ax_y.set_ylabel("y", fontsize=16, **AXIS_LABEL_STYLE)
    ax_x.set_ylabel("x", fontsize=16, **AXIS_LABEL_STYLE)

    ax_x.set_xlabel("time", fontsize=16, **AXIS_LABEL_STYLE)

    fig.suptitle(title, size="large", weight="bold")
    fig.tight_layout(pad=2)

    # ax_z.legend(loc="upper right", bbox_to_anchor=(1.0, 1.2))
    ax_z.legend(
        loc="upper right",
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        bbox_transform=fig.transFigure,
        numpoints=3,
        markerscale=11.0,
    )

    # .... Show experiment relevant information ...................................................

    if not is_pytest_run():

        if cfg.environment.obs_are_dt_derivatives:
            obs_str = "velocity, "
        else:
            obs_str = "pose, "

        if cfg.environment.obs_time_is_delta_time:
            obs_str += "delta-time"
        else:
            obs_str += "timestep"

        info_str = (
            # f"Environment math function:\n"
            # f"  {state_space_label}\n"
            f"{state_space_label}\n"
            f"  Time space granularity: {cfg.environment.time_space.granularity}\n"
            f"  Explorable space idx: {cfg.environment.explorable_space}\n"
            f"  Observations: ({obs_str})\n"
            f"{extra_info_str or ''}"
        )
        text_style = {
            "color": "#888888",
            "ha": "left",
            # "fontsize": 18
            "fontsize": 17,
        }
        fig.text(x=0.01, y=0.98, s=info_str, va="top", **text_style)
        if experiment_id is not None:
            experiment_id = f"\nExperiment id: {experiment_id}"
        else:
            experiment_id = ""
        fig.text(
            x=0.01,
            y=0.01,
            s=f"Experiment name: {cfg.experiment}{experiment_id}",
            va="bottom",
            **text_style,
        )

    plot_history_horizon_len(cfg, ax_z)

    return fig, ax_3d, ax_z, ax_x, ax_y


def _setup_1d_axis_subplot(
    axis,
    cfg: omegaconf.DictConfig,
    show_samples: bool,
    state_space: np.ndarray,
    state_space_with_noise: np.ndarray,
    time_space: np.ndarray,
    subplot_1d_interval: Optional[slice],
    selected_dimension: int,
    label: Optional[str] = None,
    skip_label: bool = False,
    show_explorable_space=True,
) -> plt.Axes:
    # state_space_with_noise_sdim = state_space_with_noise[..., selected_dimension]

    axis.plot(
        time_space,
        state_space[..., selected_dimension],
        color=COLOR_GROUND_TRUTH,
        alpha=COLOR_GROUND_TRUTH_ALPHA,
        ls="-",
        linewidth=1,
        label=label,
        zorder=4,  # Put in front of predictions
    )

    if skip_label:
        show_explorable_label_once = ""
        show_sample_label_once = ""
    else:
        if show_explorable_space:
            show_explorable_label_once = "Explorable region of the state space"
        show_sample_label_once = "Sample's"

    for each_cfg in [*cfg.environment.explorable_space]:
        explorable_interval = slice(each_cfg[0], each_cfg[1])

        if show_samples:
            axis.plot(
                time_space[explorable_interval],
                # state_space_with_noise_sdim[explorable_interval],
                state_space_with_noise[explorable_interval, selected_dimension],
                ".",
                color=COLOR_OBSERVATIONS,
                markersize=MARKERSIZE_OBSERVATIONS,
                alpha=COLOR_OBSERVATIONS_ALPHA,
                label=show_sample_label_once,
                zorder=4,  # Put in front of predictions
            )

            # Shrink noise visualization below a treshold
            # noise_alpha_treshold = np.absolute(
            #     state_space_with_noise[explorable_interval, selected_dimension]
            #     - state_space[explorable_interval, selected_dimension]
            # )
            # noise_alpha_treshold = (
            #     5
            #     * min_max_normalization(
            #         noise_alpha_treshold, scale_min=0.01, scale_max=1.0
            #     ).mean()
            # )

            noise_alpha_treshold = np.absolute(
                state_space_with_noise[explorable_interval, selected_dimension]
                - state_space[explorable_interval, selected_dimension]
            )
            noise_alpha_treshold = 1.0 * numpy_softplus(noise_alpha_treshold, beta=7.0)
            noise_alpha_treshold = 1.0 * noise_alpha_treshold.clip(max=2.0)
            noise_alpha_treshold = noise_alpha_treshold.mean()

            # Visualize noise "blur/glow" layers
            # noise_blur_cfg = [(12, 0.075), (6, 0.175), (2, 0.55)]
            # noise_blur_cfg = [(10, 0.1), (6, 0.5), (2, 0.85)]
            noise_blur_cfg = [(10, 0.1), (6, 0.5), (1.0, 0.25)]
            for each_z, (each_ms, each_a) in enumerate(noise_blur_cfg):
                each_ms *= (
                    COLOR_GROUND_TRUTH_NOISE_BLUR_MARKERSIZE * MARKERSIZE_OBSERVATIONS
                )
                each_a *= COLOR_GROUND_TRUTH_NOISE_BLUR_ALPHA * noise_alpha_treshold

                axis.plot(
                    time_space[explorable_interval],
                    state_space_with_noise[explorable_interval, selected_dimension].T,
                    ".",
                    color=COLOR_GROUND_TRUTH_NOISE_BLUR[each_z],
                    alpha=each_a,
                    markersize=each_ms,
                    zorder=each_z + 1,  # Put in behind the system ground thruth
                )

        show_sample_label_once = ""

        if show_explorable_space:
            axis.axvspan(
                xmin=time_space[explorable_interval.start],
                xmax=time_space[explorable_interval.stop - 1],
                linestyle="-",
                linewidth=1,
                color="gray",
                alpha=0.12,
                label=show_explorable_label_once,
            )
            show_explorable_label_once = ""

    axis.grid(True)
    axis.set_ylim(auto=True)
    # if not subplot_1d_interval:
    #     axis.set_xlim(*cfg.environment.time_space.bound)
    if subplot_1d_interval:
        # axis.set_xlim(subplot_1d_interval.start, subplot_1d_interval.stop)
        axis.set_xlim(
            time_space[subplot_1d_interval.start],
            time_space[subplot_1d_interval.stop],
        )

    return axis


def three_dimension_prediction_plot(
    cfg: omegaconf.DictConfig,
    time_space: np.ndarray,
    state_space_3d_base: np.ndarray,
    state_space_3d_target: np.ndarray,
    pred_mean_3d: np.ndarray,
    pred_std_3d: np.ndarray,
    pred_epi_std_3d: np.ndarray,
    title: str,
    state_space_label: str,
    show_ale_uncertainty: bool = True,
    ale_uncertainty_scaling: float = ALE_UNCERTAINTY_SCALING_DEFAULT,
    show_epi_uncertainty: bool = True,
    epi_uncertainty_scaling: float = EPI_UNCERTAINTY_SCALING_DEFAULT,
    subplot_1d_interval: Optional[slice] = None,
    figsize: tuple = (20, 8),
    figdpi: int = 50,
    extra_info_str: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
    omegaconf.OmegaConf.set_readonly(cfg, True)

    if not subplot_1d_interval:
        subplot_1d_interval = slice(0, time_space.size - 1)
    else:
        assert isinstance(subplot_1d_interval, slice)

    fig, ax_3d, ax_z, ax_x, ax_y = three_dimension_environment_space_plot(
        cfg,
        time_space=time_space,
        state_space_3d=state_space_3d_base,
        state_space_3d_with_noise=state_space_3d_target,
        title=title,
        state_space_label=state_space_label,
        subplot_1d_interval=subplot_1d_interval,
        show_samples=False,
        figsize=figsize,
        figdpi=figdpi,
        extra_info_str=extra_info_str,
        experiment_id=experiment_id,
    )

    # .... Setup 3D plot ..........................................................................
    # Show predictions
    ax_3d.plot3D(
        *pred_mean_3d[subplot_1d_interval, :].T,
        color=COLOR_PREDICTIONS,
        alpha=COLOR_LINE_3D_ALPHA,
        lw=THREE_DIM_LW,
        zorder=5,  # Put in front of the ground truth plot
    )

    # # Quick-hack to prevent 3D aspect ratio skewing
    # # Credit: https://stackoverflow.com/a/65181861
    # world_limits = ax_3d.get_w_lims()
    # ax_3d.set_box_aspect((world_limits[1] - world_limits[0], world_limits[3] - world_limits[2],
    #                       world_limits[5] - world_limits[4]))
    limits = np.array([getattr(ax_3d, f"get_{axis}lim")() for axis in "xyz"])
    ax_3d.set_box_aspect(np.ptp(limits, axis=1))

    # .... Setup xyz axes 1D subplot ..............................................................
    ax_z = _setup_1d_axis_prediction_subplot(
        ax_z,
        cfg,
        time_space=time_space[1:],
        state_space_3d_target=state_space_3d_target[1:, ...],
        pred_mean_3d=pred_mean_3d[:-1, ...],
        pred_std_3d=pred_std_3d[:-1, ...],
        pred_epi_std_3d=pred_epi_std_3d[:-1, ...],
        subplot_1d_interval=subplot_1d_interval,
        selected_dimension=DIM_Z,
        show_ale_uncertainty=show_ale_uncertainty,
        ale_uncertainty_scaling=ale_uncertainty_scaling,
        show_epi_uncertainty=show_epi_uncertainty,
        epi_uncertainty_scaling=epi_uncertainty_scaling,
    )

    ax_x = _setup_1d_axis_prediction_subplot(
        ax_x,
        cfg,
        time_space=time_space[1:],
        state_space_3d_target=state_space_3d_target[1:, ...],
        pred_mean_3d=pred_mean_3d[:-1, ...],
        pred_std_3d=pred_std_3d[:-1, ...],
        pred_epi_std_3d=pred_epi_std_3d[:-1, ...],
        subplot_1d_interval=subplot_1d_interval,
        selected_dimension=DIM_X,
        show_ale_uncertainty=show_ale_uncertainty,
        ale_uncertainty_scaling=ale_uncertainty_scaling,
        show_epi_uncertainty=show_epi_uncertainty,
        epi_uncertainty_scaling=epi_uncertainty_scaling,
    )

    ax_y = _setup_1d_axis_prediction_subplot(
        ax_y,
        cfg,
        time_space=time_space[1:],
        state_space_3d_target=state_space_3d_target[1:, ...],
        pred_mean_3d=pred_mean_3d[:-1, ...],
        pred_std_3d=pred_std_3d[:-1, ...],
        pred_epi_std_3d=pred_epi_std_3d[:-1, ...],
        subplot_1d_interval=subplot_1d_interval,
        selected_dimension=DIM_Y,
        show_ale_uncertainty=show_ale_uncertainty,
        ale_uncertainty_scaling=ale_uncertainty_scaling,
        show_epi_uncertainty=show_epi_uncertainty,
        epi_uncertainty_scaling=epi_uncertainty_scaling,
    )

    # ax_z.legend(loc="upper right", bbox_to_anchor=(1.0, 1.1))

    ax_z.legend(
        loc="upper right",
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        bbox_transform=fig.transFigure,
        numpoints=3,
        markerscale=11.0,
    )
    return fig, ax_3d, ax_z, ax_x, ax_y


def _setup_1d_axis_prediction_subplot(
    axis,
    cfg: omegaconf.DictConfig,
    time_space: np.ndarray,
    state_space_3d_target: np.ndarray,
    pred_mean_3d: np.ndarray,
    pred_std_3d: np.ndarray,
    pred_epi_std_3d: np.ndarray,
    subplot_1d_interval: slice,
    selected_dimension: int,
    show_ale_uncertainty: bool = True,
    ale_uncertainty_scaling: float = ALE_UNCERTAINTY_SCALING_DEFAULT,
    show_epi_uncertainty: bool = True,
    epi_uncertainty_scaling: float = EPI_UNCERTAINTY_SCALING_DEFAULT,
) -> plt.Axes:

    state_space_3d_target = state_space_3d_target[..., selected_dimension]
    pred_mean_3d = pred_mean_3d[..., selected_dimension]
    pred_std_3d = pred_std_3d[..., selected_dimension]
    pred_epi_std_3d = pred_epi_std_3d[..., selected_dimension]
    pred_ale_std_3d = pred_std_3d - pred_epi_std_3d

    epi_upper_bound = pred_mean_3d + pred_epi_std_3d * epi_uncertainty_scaling
    epi_lower_bound = pred_mean_3d - pred_epi_std_3d * epi_uncertainty_scaling

    ale_upper_bound = epi_upper_bound + pred_ale_std_3d * ale_uncertainty_scaling
    ale_lower_bound = epi_lower_bound - pred_ale_std_3d * ale_uncertainty_scaling

    # .... Observations ...........................................................................
    axis.plot(
        time_space,
        state_space_3d_target,
        ".",
        color=COLOR_OBSERVATIONS,
        markersize=MARKERSIZE_OBSERVATIONS,
        alpha=COLOR_OBSERVATIONS_ALPHA,
        zorder=4,  # Put in front of predictions
        label="Observations",
    )

    # .... Predictions ............................................................................
    axis.plot(
        time_space,
        pred_mean_3d,
        ".",
        color=COLOR_PREDICTIONS,
        markersize=MARKERSIZE_PREDICTIONS,
        alpha=COLOR_PREDICTIONS_ALPHA,
        zorder=3,  # Put behind the ground truth plot but in front of fills
        label="Predictions",
    )

    # .... Aleatoric uncertainty standard deviation ...............................................
    if show_ale_uncertainty and show_epi_uncertainty:
        axis.fill_between(
            time_space,
            ale_upper_bound,
            epi_upper_bound,
            color=COLOR_ALE,
            alpha=COLOR_ALE_ALPHA,
            linewidth=0,
            antialiased=True,
            zorder=1,  # Put behind EPI fills
            label=f"Aleatoric uncertainty ({ale_uncertainty_scaling}" + r"$\sigma$)",
        )
        axis.fill_between(
            time_space,
            epi_lower_bound,
            ale_lower_bound,
            color=COLOR_ALE,
            linewidth=0,
            alpha=COLOR_ALE_ALPHA,
            antialiased=True,
            zorder=1,  # Put behind EPI fills
        )
    elif show_ale_uncertainty and not show_epi_uncertainty:
        axis.fill_between(
            time_space,
            pred_mean_3d + pred_std_3d * ale_uncertainty_scaling,
            pred_mean_3d - pred_std_3d * ale_uncertainty_scaling,
            color=COLOR_ALE,
            alpha=COLOR_ALE_ALPHA,
            linewidth=0,
            antialiased=True,
            zorder=1,
            label=f"Aleatoric uncertainty ({ale_uncertainty_scaling}" + r"$\sigma$)",
        )

    # .... Epistemic uncertainty standard deviation ...............................................
    if show_epi_uncertainty:
        axis.fill_between(
            time_space,
            epi_upper_bound,
            epi_lower_bound,
            color=COLOR_EPI,
            alpha=COLOR_EPI_ALPHA,
            linewidth=0,
            antialiased=True,
            zorder=2,  # Put behind predictions but in front of EPI fills
            label=f"Epistemic uncertainty ({epi_uncertainty_scaling}" + r"$\sigma$)",
        )
        # Alt setup
        # axis.fill_between(
        #     time_space,
        #     epi_upper_bound,
        #     pred_mean_3d,
        #     color=COLOR_EPI,
        #     alpha=COLOR_EPI_ALPHA,
        #     label="Uncertainty (epistemic)",
        #     linewidth=0,
        #     zorder=2,  # Put behind predictions but in front of EPI fills
        #     antialiased=True,
        # )
        # axis.fill_between(
        #     time_space,
        #     pred_mean_3d,
        #     epi_lower_bound,
        #     color=COLOR_EPI,
        #     alpha=COLOR_EPI_ALPHA,
        #     linewidth=0,
        #     zorder=2,  # Put behind predictions but in front of EPI fills
        #     antialiased=True,
        # )

    # .... Subplot limits .........................................................................
    axis.set_ylim(auto=True)
    # if not subplot_1d_interval:
    #     axis.set_xlim(*cfg.environment.time_space.bound)
    if subplot_1d_interval:
        # axis.set_xlim(subplot_1d_interval.start, subplot_1d_interval.stop)
        axis.set_xlim(
            time_space[subplot_1d_interval.start],
            time_space[subplot_1d_interval.stop - 1],
        )

    return axis


def plot_history_horizon_len(cfg: omegaconf.DictConfig, ax_z: plt.Axes) -> plt.Axes:
    # Position on plot from top left corner
    v_offset = 200
    h_offset = 100

    granularity = cfg.environment.time_space.granularity
    history_len = cfg.ms_model.history_len
    horizon_len = cfg.ms_model.horizon_len
    bar_height = 50

    x_plot_scale = 1 / granularity
    y_plot_scale = 1 / (ax_z.get_ylim()[1] - ax_z.get_ylim()[0])

    # Note: axis units are in 0:1
    history_start = x_plot_scale * h_offset
    history_end = history_start + x_plot_scale * history_len
    ax_z.axhspan(
        ymax=ax_z.get_ylim()[1] - y_plot_scale * v_offset,
        ymin=ax_z.get_ylim()[1] - y_plot_scale * (v_offset - bar_height),
        xmin=history_start,
        xmax=history_end,
        # color="whitesmoke",
        # color="white",
        # color="gainsboro",
        color="dimgray",
    )
    horizon_start = history_end
    horizon_end = horizon_start + x_plot_scale * horizon_len
    ax_z.axhspan(
        ymax=ax_z.get_ylim()[1] - y_plot_scale * v_offset,
        ymin=ax_z.get_ylim()[1] - y_plot_scale * (v_offset - bar_height),
        xmin=horizon_start,
        xmax=horizon_end,
        # color="silver",
        color="darkgray",
    )
    return ax_z
