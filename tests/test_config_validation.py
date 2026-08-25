import copy

import pytest
from pydantic import ValidationError

from juniper_auto.config import ArchitectureConfig, load_architecture_config


def test_valid_sparse_configuration_accepted(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    assert cfg.architecture_id == "ja150m-v0.1"
    assert cfg.kind == "sparse"


def test_valid_dense_configuration_accepted(dense_config_path):
    cfg = load_architecture_config(dense_config_path)
    assert cfg.architecture_id == "ja150m-v0.1-dense"
    assert cfg.kind == "dense"


def test_layer_partition_is_exactly_20_layers(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    assert cfg.core.n_layers == 20
    assert len(set(cfg.core.dense_layers) | set(cfg.core.moe_layers)) == 20


def test_dense_positions_are_correct(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    assert cfg.core.dense_layers == [1, 5, 10, 15, 20]


def test_moe_positions_are_correct(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    assert cfg.core.moe_layers == [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
    assert len(cfg.core.moe_layers) == 15


@pytest.fixture
def valid_sparse_dict(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    return cfg.model_dump(mode="json")


def _rebuild(d):
    return ArchitectureConfig.model_validate(d)


class TestMalformedConfigurationRejected:
    def test_duplicate_layer_numbers(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["core"]["moe_layers"] = d["core"]["moe_layers"] + [1]  # 1 is already a dense layer
        with pytest.raises(ValidationError, match="duplicate|overlap"):
            _rebuild(d)

    def test_dense_moe_overlap(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["core"]["moe_layers"] = [1] + d["core"]["moe_layers"][1:]
        with pytest.raises(ValidationError, match="duplicate|overlap"):
            _rebuild(d)

    def test_missing_layer(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["core"]["moe_layers"] = d["core"]["moe_layers"][1:]  # drop layer 2
        with pytest.raises(ValidationError, match="missing"):
            _rebuild(d)

    def test_out_of_range_layer(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["core"]["moe_layers"] = d["core"]["moe_layers"] + [21]
        with pytest.raises(ValidationError, match="out-of-range"):
            _rebuild(d)

    def test_invalid_expert_count(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["moe"]["n_routed_experts"] = 0
        with pytest.raises(ValidationError):
            _rebuild(d)

    def test_impossible_top_k(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["moe"]["top_k"] = 99
        with pytest.raises(ValidationError, match="top_k"):
            _rebuild(d)

    def test_invalid_head_divisibility(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["attention"]["n_kv_heads"] = 3  # 8 % 3 != 0
        with pytest.raises(ValidationError, match="divisible"):
            _rebuild(d)

    def test_head_dim_mismatch_with_d_model(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["attention"]["head_dim"] = 63
        with pytest.raises(ValidationError, match="d_model"):
            _rebuild(d)

    def test_invalid_context_length(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["attention"]["context_length"] = 0
        with pytest.raises(ValidationError):
            _rebuild(d)

    def test_token_dropping_rejected_for_v01(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["moe"]["token_dropping_allowed"] = True
        with pytest.raises(ValidationError, match="dropless"):
            _rebuild(d)

    def test_dropless_false_rejected_for_v01(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["moe"]["dropless"] = False
        with pytest.raises(ValidationError, match="dropless"):
            _rebuild(d)

    def test_invalid_frozen_architecture_id_kind_mismatch(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["architecture_id"] = "ja150m-v0.1-dense"  # known id, wrong kind for this payload
        with pytest.raises(ValidationError, match="invalid frozen architecture id"):
            _rebuild(d)

    def test_sparse_kind_requires_moe_section(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["architecture_id"] = "some-future-arch"
        d["kind"] = "sparse"
        d["moe"] = None
        with pytest.raises(ValidationError, match="moe"):
            _rebuild(d)

    def test_missing_required_field_rejected(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        del d["core"]["d_model"]
        with pytest.raises(ValidationError):
            _rebuild(d)

    def test_extra_field_rejected(self, valid_sparse_dict):
        d = copy.deepcopy(valid_sparse_dict)
        d["totally_unexpected_field"] = 123
        with pytest.raises(ValidationError):
            _rebuild(d)

    def test_dense_kind_must_not_declare_moe_layers(self, valid_sparse_dict, dense_config_path):
        cfg = load_architecture_config(dense_config_path)
        d = cfg.model_dump(mode="json")
        d["core"]["moe_layers"] = [2]
        d["core"]["dense_layers"] = [x for x in d["core"]["dense_layers"] if x != 2]
        with pytest.raises(ValidationError, match="moe_layers"):
            _rebuild(d)
