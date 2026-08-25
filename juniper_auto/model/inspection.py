"""Introspection utilities over an actually-instantiated `JuniperAutoModel`.

This is deliberately independent of `juniper_auto.accounting.parameter_count`
(Method A, which counts from the *config*): everything here walks the real
`nn.Module` tree and sums real `Parameter.numel()` values (Method B), so
the two verification paths cannot share a bug. Used by tests, by
`scripts/validate_phase1.py`, and by the self-review module-structure
audit -- a config can be correct while the code that builds modules from
it is wrong, so this must inspect what actually executes.
"""

from __future__ import annotations

import torch
from torch import nn

from juniper_auto.model.attention import GroupedQueryAttention
from juniper_auto.model.block import DenseBlock, MoEBlock
from juniper_auto.model.model import JuniperAutoModel


def total_parameters(model: nn.Module) -> int:
    """sum(p.numel() for p in model.parameters()) -- relies on PyTorch's
    default de-duplication-by-object-identity in `Module.parameters()`, so
    a tied weight (e.g. embedding/LM-head) is counted exactly once."""
    return sum(p.numel() for p in model.parameters())


def pytorch_parameter_breakdown(model: JuniperAutoModel) -> dict[str, int]:
    """Category totals computed directly from the instantiated module tree.
    Keys match `juniper_auto.accounting.parameter_count.ParameterBreakdown`
    so the two independent counting methods can be compared field by field.
    """
    embeddings = model.embedding.weight.numel()

    attention = 0
    qk_norms = 0
    dense_ffns = 0
    routed_experts = 0
    shared_experts = 0
    routers = 0
    block_norms = 0

    for block in model.layers:
        attn = block.attention
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            attention += proj.weight.numel()
            if proj.bias is not None:
                attention += proj.bias.numel()
        if attn.qk_norm:
            qk_norms += attn.q_norm.weight.numel() + attn.k_norm.weight.numel()

        block_norms += block.attention_norm.weight.numel() + block.ffn_norm.weight.numel()

        if isinstance(block, DenseBlock):
            dense_ffns += sum(p.numel() for p in block.ffn.parameters())
        elif isinstance(block, MoEBlock):
            moe = block.moe
            routers += sum(p.numel() for p in moe.router.parameters())
            for expert in moe.routed_experts:
                routed_experts += sum(p.numel() for p in expert.parameters())
            shared_experts += sum(p.numel() for p in moe.shared_expert.parameters())
        else:
            raise TypeError(f"unexpected block type: {type(block)!r}")

    final_norm = model.final_norm.weight.numel()

    breakdown = {
        "embeddings": embeddings,
        "attention": attention,
        "dense_ffns": dense_ffns,
        "routed_experts": routed_experts,
        "shared_experts": shared_experts,
        "routers": routers,
        "qk_norms": qk_norms,
        "block_norms": block_norms,
        "final_norm": final_norm,
    }
    breakdown["total"] = sum(breakdown.values())
    return breakdown


def verify_weight_tying(model: JuniperAutoModel) -> bool:
    return model.embedding.weight is model.lm_head.weight


def bias_audit(model: nn.Module) -> list[str]:
    """Returns the dotted names of any Linear/Embedding parameter that has a
    non-None bias. Empty list means the "no unintended bias" requirement
    holds."""
    offenders = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            offenders.append(f"{name}.bias")
    return offenders


def dropout_audit(model: nn.Module) -> list[str]:
    """Returns the dotted names of any `nn.Dropout` submodule with nonzero
    probability. The reference implementation does not instantiate any
    `nn.Dropout` modules at all (every frozen dropout config value is
    0.0), so this should always return an empty list -- that absence is
    itself the audited property, not merely a fallback."""
    offenders = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) and module.p != 0.0:
            offenders.append(f"{name} (p={module.p})")
    return offenders


def layer_placement_report(model: JuniperAutoModel) -> list[dict]:
    report = []
    for index, (kind, block) in enumerate(zip(model.layer_kinds, model.layers), start=1):
        entry = {"layer": index, "kind": kind, "module_type": type(block).__name__}
        if isinstance(block, MoEBlock):
            entry["n_routed_experts"] = len(block.moe.routed_experts)
            entry["n_shared_experts"] = 1
            entry["top_k"] = block.moe.top_k
        report.append(entry)
    return report


def qk_norm_parameter_count(model: JuniperAutoModel) -> int:
    total = 0
    for block in model.layers:
        attn = block.attention
        if attn.qk_norm:
            total += attn.q_norm.weight.numel() + attn.k_norm.weight.numel()
    return total
