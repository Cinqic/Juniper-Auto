from juniper_auto.config import load_architecture_config
from juniper_auto.foundation import run_foundation_probe


def test_foundation_probe_deterministic_under_same_seed_cpu(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    r1 = run_foundation_probe(cfg, seed=42, device="cpu")
    r2 = run_foundation_probe(cfg, seed=42, device="cpu")
    assert r1.output_checksum == r2.output_checksum
    assert r1.output_shape == r2.output_shape


def test_foundation_probe_differs_under_different_seed_cpu(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    r1 = run_foundation_probe(cfg, seed=1, device="cpu")
    r2 = run_foundation_probe(cfg, seed=2, device="cpu")
    assert r1.output_checksum != r2.output_checksum


def test_seed_report_never_silently_defaults(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    result = run_foundation_probe(cfg, seed=7, device="cpu")
    assert result.seed_report.seed == 7
    assert result.seed_report.python_random_seeded
    assert result.seed_report.numpy_seeded
    assert result.seed_report.torch_cpu_seeded


def test_foundation_probe_requires_explicit_seed_type():
    import pytest

    from juniper_auto.util.seed import apply_seed

    with pytest.raises(TypeError):
        apply_seed("not-an-int")  # type: ignore[arg-type]


def test_foundation_probe_runs_on_cpu_without_gpu_requirement(sparse_config_path):
    """No hidden GPU requirement: explicitly requesting cpu must work
    regardless of whether CUDA happens to be available on this machine."""
    cfg = load_architecture_config(sparse_config_path)
    result = run_foundation_probe(cfg, seed=99, device="cpu")
    assert result.device == "cpu"
