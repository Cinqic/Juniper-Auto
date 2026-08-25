import io
import json
import logging

from juniper_auto.util.logging import LogContext, get_logger, log_event


def test_log_event_emits_valid_json_with_required_fields():
    stream = io.StringIO()
    logger = logging.getLogger("test.juniper_auto.logging.unique1")
    logger.handlers.clear()
    from juniper_auto.util.logging import _JsonFormatter

    handler = logging.StreamHandler(stream)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    ctx = LogContext(
        phase="phase-0",
        run_id="run-1",
        git_commit="abc123",
        config_id="ja150m-v0.1",
        architecture_id="ja150m-v0.1",
        seed=42,
        env_id="py3.12-torch2.13-cpu-only",
    )
    log_event(logger, logging.INFO, "test.event", ctx)

    line = stream.getvalue().strip()
    payload = json.loads(line)  # must be valid JSON, one line

    assert payload["event"] == "test.event"
    assert payload["level"] == "INFO"
    assert "timestamp" in payload
    assert payload["phase"] == "phase-0"
    assert payload["run_id"] == "run-1"
    assert payload["git_commit"] == "abc123"
    assert payload["config_id"] == "ja150m-v0.1"
    assert payload["architecture_id"] == "ja150m-v0.1"
    assert payload["seed"] == 42
    assert payload["env_id"] == "py3.12-torch2.13-cpu-only"


def test_log_context_omits_none_fields():
    ctx = LogContext(phase="phase-0")
    d = ctx.as_dict()
    assert d == {"phase": "phase-0"}
    assert "run_id" not in d
    assert "seed" not in d


def test_log_context_includes_extra_fields():
    ctx = LogContext(phase="phase-0", extra={"device": "cpu", "checksum": 1.5})
    d = ctx.as_dict()
    assert d["device"] == "cpu"
    assert d["checksum"] == 1.5


def test_get_logger_does_not_duplicate_handlers_on_repeated_calls():
    logger1 = get_logger("test.juniper_auto.logging.unique2")
    logger2 = get_logger("test.juniper_auto.logging.unique2")
    assert logger1 is logger2
    assert len(logger1.handlers) == 1


def test_foundation_probe_emits_start_and_complete_log_events(sparse_config_path, capsys):
    """End-to-end: running the foundation probe actually exercises the
    logging module, not just tensor math."""
    from juniper_auto.config import load_architecture_config
    from juniper_auto.foundation import run_foundation_probe

    cfg = load_architecture_config(sparse_config_path)
    logger = get_logger("juniper_auto.foundation.probe")
    stream = io.StringIO()
    from juniper_auto.util.logging import _JsonFormatter

    handler = logging.StreamHandler(stream)
    handler.setFormatter(_JsonFormatter())
    logger.handlers = [handler]

    run_foundation_probe(cfg, seed=3, device="cpu")

    lines = [line for line in stream.getvalue().strip().splitlines() if line]
    assert len(lines) == 2
    events = [json.loads(line)["event"] for line in lines]
    assert events == ["foundation_probe.start", "foundation_probe.complete"]
