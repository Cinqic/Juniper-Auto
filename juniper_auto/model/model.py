"""The Juniper Auto reference model: shared embeddings/attention/norm/RoPE
infrastructure with a per-layer dense-anchor or MoE FFN, selected by the
architecture config's `core.dense_layers` / `core.moe_layers` partition.
Used unmodified for both `ja150m-v0.1` (sparse) and `ja150m-v0.1-dense`
(dense control, whose `moe_layers` is empty so every block is a
`DenseBlock`).

    ArchitectureConfig -> build_model(cfg) -> JuniperAutoModel
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from juniper_auto.config.schema import ArchitectureConfig
from juniper_auto.model.attention import GroupedQueryAttention
from juniper_auto.model.block import DenseBlock, MoEBlock
from juniper_auto.model.ffn import SwiGLU
from juniper_auto.model.losses import causal_lm_loss
from juniper_auto.model.moe import MoELayer, MoEDiagnostics
from juniper_auto.model.norm import RMSNorm


@dataclass
class ModelOutput:
    logits: torch.Tensor  # [batch, seq_len, vocab], FP32
    loss: torch.Tensor | None  # lm_loss + weighted aux losses (sparse) or lm_loss (dense)
    lm_loss: torch.Tensor | None
    load_balance_loss: torch.Tensor  # weighted; exactly 0.0 for a dense model
    router_z_loss: torch.Tensor  # weighted; exactly 0.0 for a dense model
    load_balance_loss_raw: torch.Tensor  # unweighted, averaged across MoE layers
    router_z_loss_raw: torch.Tensor  # unweighted, averaged across MoE layers
    diagnostics: list[MoEDiagnostics | None] | None  # per layer, only if requested


class JuniperAutoModel(nn.Module):
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.embeddings.kind != "learned":
            raise ValueError(f"unsupported embeddings.kind: {cfg.embeddings.kind!r} (only 'learned' is implemented)")
        if not cfg.normalization.final_norm:
            raise ValueError("normalization.final_norm must be true (the final norm is applied unconditionally)")
        if cfg.normalization.reduction_dtype != "fp32":
            raise ValueError("normalization.reduction_dtype must be 'fp32' (RMSNorm reductions are forced to FP32)")
        if cfg.normalization.layernorm_bias:
            raise ValueError("normalization.layernorm_bias=true is unsupported (RMSNorm has no bias)")
        nonzero_dropout = {
            name: value
            for name, value in cfg.dropout.model_dump().items()
            if value != 0.0
        }
        if nonzero_dropout:
            raise ValueError(
                f"nonzero dropout is not implemented (no nn.Dropout modules exist in this model): {nonzero_dropout}"
            )

        self.embedding = nn.Embedding(cfg.embeddings.vocab_size, cfg.embeddings.dim)
        self.embedding_scale = cfg.embeddings.embedding_scale
        self.context_length = cfg.attention.context_length

        layer_kind: dict[int, str] = {}
        for layer_num in cfg.core.dense_layers:
            layer_kind[layer_num] = "dense"
        for layer_num in cfg.core.moe_layers:
            layer_kind[layer_num] = "moe"

        blocks = [
            DenseBlock(cfg) if layer_kind[i] == "dense" else MoEBlock(cfg)
            for i in range(1, cfg.core.n_layers + 1)
        ]
        self.layers = nn.ModuleList(blocks)
        self.layer_kinds = [layer_kind[i] for i in range(1, cfg.core.n_layers + 1)]

        self.final_norm = RMSNorm(cfg.core.d_model, eps=cfg.normalization.epsilon)

        self.lm_head = nn.Linear(cfg.embeddings.dim, cfg.embeddings.vocab_size, bias=cfg.embeddings.output_bias)
        if cfg.embeddings.tie_lm_head:
            self.lm_head.weight = self.embedding.weight
        self.logit_softcap = cfg.embeddings.logit_softcap

        self.load_balance_coefficient = cfg.moe.load_balance_loss_coefficient if cfg.moe is not None else 0.0
        self.router_z_coefficient = cfg.moe.router_z_loss_coefficient if cfg.moe is not None else 0.0

        # Optional, off by default -- see juniper_auto.model.model.JuniperAutoModel.set_gradient_checkpointing.
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enables/disables per-block activation (gradient) checkpointing:
        recomputes each block's forward during backward instead of holding
        its activations, trading compute for peak activation memory. Only
        takes effect in training mode, and is incompatible with
        `return_diagnostics=True` (per-layer MoE diagnostics would have to
        be recomputed and discarded, defeating the purpose)."""
        self.gradient_checkpointing = enabled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}")
        batch, seq_len = input_ids.shape
        if batch == 0 or seq_len == 0:
            raise ValueError("input_ids must contain at least one batch row and one token")
        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds the supported context length {self.context_length}; "
                "the future context target is not a validated capability"
            )
        for name, value in (("attention_mask", attention_mask), ("labels", labels), ("position_ids", position_ids)):
            if value is not None and value.shape != input_ids.shape:
                raise ValueError(
                    f"{name} must have the same [batch, seq_len] shape as input_ids "
                    f"({tuple(input_ids.shape)}), got {tuple(value.shape)}"
                )
            if value is not None and value.device != input_ids.device:
                raise ValueError(f"{name} must be on the same device as input_ids")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)

        key_valid_mask = attention_mask.to(torch.bool) if attention_mask is not None else None

        x = self.embedding(input_ids) * self.embedding_scale

        load_balance_raws: list[torch.Tensor] = []
        router_z_raws: list[torch.Tensor] = []
        layer_diagnostics: list[MoEDiagnostics | None] | None = [] if return_diagnostics else None

        use_checkpointing = self.gradient_checkpointing and self.training
        if use_checkpointing and return_diagnostics:
            raise ValueError("gradient checkpointing does not support return_diagnostics=True")

        for block in self.layers:
            if use_checkpointing:
                x, lb_raw, z_raw, diag = torch_checkpoint(
                    block, x, position_ids, key_valid_mask, False, use_reentrant=False
                )
            else:
                x, lb_raw, z_raw, diag = block(
                    x, position_ids, key_valid_mask, return_diagnostics=return_diagnostics
                )
            if lb_raw is not None:
                load_balance_raws.append(lb_raw)
            if z_raw is not None:
                router_z_raws.append(z_raw)
            if return_diagnostics:
                layer_diagnostics.append(diag)

        x = self.final_norm(x)
        logits = self.lm_head(x).to(torch.float32)
        if self.logit_softcap is not None:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if load_balance_raws:
            load_balance_raw = torch.stack(load_balance_raws).mean()
        else:
            load_balance_raw = logits.new_zeros(())
        if router_z_raws:
            router_z_raw = torch.stack(router_z_raws).mean()
        else:
            router_z_raw = logits.new_zeros(())

        load_balance_loss = self.load_balance_coefficient * load_balance_raw
        router_z_loss = self.router_z_coefficient * router_z_raw

        loss = None
        lm_loss = None
        if labels is not None:
            lm_loss = causal_lm_loss(logits, labels)
            loss = lm_loss + load_balance_loss + router_z_loss

        return ModelOutput(
            logits=logits,
            loss=loss,
            lm_loss=lm_loss,
            load_balance_loss=load_balance_loss,
            router_z_loss=router_z_loss,
            load_balance_loss_raw=load_balance_raw,
            router_z_loss_raw=router_z_raw,
            diagnostics=layer_diagnostics,
        )


def _init_normal(
    weight: torch.Tensor, mean: float, std: float, generator: torch.Generator | None
) -> None:
    with torch.no_grad():
        weight.normal_(mean=mean, std=std, generator=generator)


def initialize_weights(
    model: JuniperAutoModel,
    cfg: ArchitectureConfig,
    generator: torch.Generator | None = None,
) -> None:
    """Applies the frozen v0.1 initialization policy to an already-constructed
    model, in a fixed traversal order so a given `generator` state always
    produces the same final weights.

    - General projection matrices (Q/K/V, FFN gate/up, expert gate/up):
      N(0, base_std^2).
    - Residual-output projections (attention o_proj; dense/routed/shared
      FFN down_proj): N(0, residual_output_projection_std^2) -- the smaller
      scaled-init std used to keep residual-stream variance bounded across
      layers.
    - Router: N(0, router_std^2).
    - Embedding (and, via tying, the LM head): N(0, embedding_std^2).
    - RMSNorm/QK-Norm scale vectors are left at the ones-initialization set
      by `RMSNorm.__init__` -- not touched here.
    - No bias parameters exist for the frozen v0.1 configs, so there is no
      bias-init branch.
    """
    init_cfg = cfg.initialization
    for module in model.modules():
        if isinstance(module, GroupedQueryAttention):
            _init_normal(module.q_proj.weight, init_cfg.mean, init_cfg.base_std, generator)
            _init_normal(module.k_proj.weight, init_cfg.mean, init_cfg.base_std, generator)
            _init_normal(module.v_proj.weight, init_cfg.mean, init_cfg.base_std, generator)
            _init_normal(module.o_proj.weight, init_cfg.mean, init_cfg.residual_output_projection_std, generator)
        elif isinstance(module, SwiGLU):
            _init_normal(module.gate_proj.weight, init_cfg.mean, init_cfg.base_std, generator)
            _init_normal(module.up_proj.weight, init_cfg.mean, init_cfg.base_std, generator)
            _init_normal(module.down_proj.weight, init_cfg.mean, init_cfg.residual_output_projection_std, generator)
        elif isinstance(module, MoELayer):
            _init_normal(module.router.weight, init_cfg.mean, init_cfg.router_std, generator)

    _init_normal(model.embedding.weight, init_cfg.mean, init_cfg.embedding_std, generator)


def build_model(
    cfg: ArchitectureConfig,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> JuniperAutoModel:
    """Construct a JuniperAutoModel from an ArchitectureConfig and apply the
    frozen initialization policy. Construction and initialization always
    happen on CPU (so initialization is reproducible independent of CUDA
    RNG availability/behavior); `device`/`dtype` moves are applied last.

    `seed`, if given, drives an explicit local `torch.Generator` for
    initialization -- this does not touch global RNG state, so it composes
    safely with `juniper_auto.util.seed.apply_seed` or with no seeding at
    all (in which case initialization consumes ambient global RNG state).
    """
    # PyTorch module constructors perform their own default initialization.
    # When an explicit seed requests local-generator isolation, protect the
    # ambient CPU RNG from that otherwise-observable constructor work; the
    # real frozen initialization below is then driven solely by `generator`.
    if seed is not None:
        with torch.random.fork_rng(devices=[]):
            model = JuniperAutoModel(cfg)
    else:
        model = JuniperAutoModel(cfg)
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    initialize_weights(model, cfg, generator=generator)

    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    return model
