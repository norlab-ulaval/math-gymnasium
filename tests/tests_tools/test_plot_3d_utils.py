# coding=utf-8
import omegaconf
import pytest

from math_gymnasium.tools import plot_3d_utils


@plot_3d_utils._cfg_readonly_scope
def _fake_plot(cfg: omegaconf.DictConfig, fail: bool = False):
    readonly_inside = omegaconf.OmegaConf.is_readonly(cfg)
    if fail:
        raise ValueError("boom")
    return readonly_inside


@pytest.mark.parametrize("caller_flag", [None, False, True])
def test_cfg_readonly_scope_restores_caller_flag(caller_flag):
    cfg = omegaconf.OmegaConf.create({"a": {"b": 1}})
    omegaconf.OmegaConf.set_readonly(cfg, caller_flag)

    assert _fake_plot(cfg) is True
    assert cfg._get_node_flag("readonly") is caller_flag


def test_cfg_readonly_scope_restores_on_exception_and_allows_later_writes():
    cfg = omegaconf.OmegaConf.create({"a": {"b": 1}})
    with pytest.raises(ValueError, match="boom"):
        _fake_plot(cfg, fail=True)
    assert not omegaconf.OmegaConf.is_readonly(cfg)

    _fake_plot(cfg)
    with omegaconf.open_dict(cfg.a):  # e.g., a later model/trainer rebuild writing to cfg
        cfg.a.c = 2
    assert cfg.a.c == 2


@pytest.mark.parametrize(
    "plot_fct",
    [
        plot_3d_utils.three_dimension_environment_space_plot,
        plot_3d_utils.three_dimension_prediction_plot,
    ],
)
def test_public_3d_plot_functions_are_readonly_scoped(plot_fct):
    # Both helpers must only flag cfg read-only for their own duration (no leaked global flag).
    assert hasattr(plot_fct, "__wrapped__")
