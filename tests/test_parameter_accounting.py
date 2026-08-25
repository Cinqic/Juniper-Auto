from juniper_auto.accounting import standard_active_parameter_breakdown, total_parameter_breakdown
from juniper_auto.config import load_architecture_config


def test_sparse_total_is_exact(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    breakdown = total_parameter_breakdown(cfg)
    assert breakdown.total == 150_031_360


def test_sparse_total_breakdown_matches_specification(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    breakdown = total_parameter_breakdown(cfg)
    assert breakdown.embeddings == 18_874_368
    assert breakdown.attention == 13_107_200
    assert breakdown.dense_ffns == 11_796_480
    assert breakdown.routed_experts == 94_371_840
    assert breakdown.shared_experts == 11_796_480
    assert breakdown.routers == 61_440
    assert breakdown.qk_norms == 2_560
    assert breakdown.block_norms == 20_480
    assert breakdown.final_norm == 512


def test_sparse_standard_active_is_exact(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    breakdown = standard_active_parameter_breakdown(cfg)
    assert breakdown.total == 79_252_480


def test_dense_control_total_is_exact(dense_config_path):
    cfg = load_architecture_config(dense_config_path)
    breakdown = total_parameter_breakdown(cfg)
    assert breakdown.total == 79_191_040


def test_dense_control_has_no_moe_contribution(dense_config_path):
    cfg = load_architecture_config(dense_config_path)
    breakdown = total_parameter_breakdown(cfg)
    assert breakdown.routed_experts == 0
    assert breakdown.shared_experts == 0
    assert breakdown.routers == 0


def test_active_equals_total_for_dense_control(dense_config_path):
    cfg = load_architecture_config(dense_config_path)
    total = total_parameter_breakdown(cfg).total
    active = standard_active_parameter_breakdown(cfg).total
    assert total == active


def test_accounting_changes_if_config_changes(sparse_config_path):
    """Sanity check that the accounting is actually derived from the config
    and not hard-coded: perturbing top_k must change the active count."""
    cfg = load_architecture_config(sparse_config_path)
    perturbed = cfg.model_copy(deep=True, update={"moe": cfg.moe.model_copy(update={"top_k": 1})})
    baseline_active = standard_active_parameter_breakdown(cfg).total
    perturbed_active = standard_active_parameter_breakdown(perturbed).total
    assert perturbed_active != baseline_active
    assert perturbed_active < baseline_active
