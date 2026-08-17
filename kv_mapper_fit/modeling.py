"""Model loading, attention capture, and RoPE utilities.

The capture path registers a custom attention implementation with
transformers' ALL_ATTENTION_FUNCTIONS registry. At call time it receives the
*post-RoPE* queries and keys and the values for every layer, records them,
then delegates to SDPA. This is architecture-agnostic for the decoder
families the paper targets (Llama / Qwen / Mistral style GQA with per-layer
RoPE), and correctly includes per-head QK-norm (e.g. Qwen3) because those run
before the attention function is invoked.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RECORDER_IMPL = "kvmf_recorder"


# --------------------------------------------------------------------------
# Device / dtype resolution
# --------------------------------------------------------------------------

def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype != "auto":
        return getattr(torch, dtype)
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


# --------------------------------------------------------------------------
# Attention recorder
# --------------------------------------------------------------------------

class AttentionRecorder:
    """Records post-RoPE Q/K and V per layer during a forward pass.

    Tensors are stored as [batch, n_heads, seq, head_dim] (Q uses the query
    head count; K/V use the KV head count, un-replicated).
    """

    def __init__(self) -> None:
        self.queries: Dict[int, torch.Tensor] = {}
        self.keys: Dict[int, torch.Tensor] = {}
        self.values: Dict[int, torch.Tensor] = {}
        self.record_queries = False

    def clear(self) -> None:
        self.queries.clear()
        self.keys.clear()
        self.values.clear()


_ACTIVE_RECORDER: Optional[AttentionRecorder] = None


def _recorder_attention(module, query, key, value, attention_mask, **kwargs):
    """Custom attention fn: record tensors, then run the eager/sdpa path."""
    global _ACTIVE_RECORDER
    rec = _ACTIVE_RECORDER
    if rec is not None:
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            raise RuntimeError(
                "Attention module has no layer_idx; unsupported architecture."
            )
        rec.keys[layer_idx] = key.detach().to("cpu", torch.float32)
        rec.values[layer_idx] = value.detach().to("cpu", torch.float32)
        if rec.record_queries:
            rec.queries[layer_idx] = query.detach().to("cpu", torch.float32)

    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    return sdpa_attention_forward(
        module, query, key, value, attention_mask, **kwargs
    )


def _register_recorder() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if RECORDER_IMPL not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS[RECORDER_IMPL] = _recorder_attention


# --------------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------------

class CapturableModel:
    """A causal LM plus the machinery to capture per-layer post-RoPE K/Q and V."""

    def __init__(self, name_or_path: str, device: torch.device, dtype: torch.dtype,
                 trust_remote_code: bool = False):
        self.name = name_or_path
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path, torch_dtype=dtype, trust_remote_code=trust_remote_code
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=trust_remote_code
        )
        cfg = self.model.config
        self.num_layers: int = cfg.num_hidden_layers
        self.num_kv_heads: int = getattr(
            cfg, "num_key_value_heads", cfg.num_attention_heads
        )
        self.num_q_heads: int = cfg.num_attention_heads
        self.head_dim: int = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
        _register_recorder()
        self._set_attn_impl(RECORDER_IMPL)

    def _set_attn_impl(self, impl: str) -> None:
        try:
            self.model.set_attn_implementation(impl)
        except (AttributeError, ValueError):
            self.model.config._attn_implementation = impl
            for sub in self.model.modules():
                if hasattr(sub, "config"):
                    sub.config._attn_implementation = impl

    # ------------------------------------------------------------------
    def rope_cos_sin(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for the model's RoPE at the given positions.

        positions: [batch, seq] int64. Returns cos, sin as [batch, seq, head_dim]
        float32 on CPU. Uses the model's own rotary embedding module so RoPE
        scaling variants (e.g. Llama 3.1) are handled.
        """
        rotary = self._find_rotary()
        pos = positions.to(self.device)
        probe = torch.zeros(1, dtype=self.model.dtype, device=self.device)
        cos, sin = rotary(probe, pos)
        return cos.to("cpu", torch.float32), sin.to("cpu", torch.float32)

    def _find_rotary(self):
        base = getattr(self.model, "model", self.model)
        rotary = getattr(base, "rotary_emb", None)
        if rotary is None:
            # older layouts keep it on the first attention module
            layer0 = base.layers[0]
            rotary = getattr(layer0.self_attn, "rotary_emb", None)
        if rotary is None:
            raise RuntimeError(f"Could not locate rotary embedding on {self.name}")
        return rotary

    # ------------------------------------------------------------------
    @torch.no_grad()
    def capture(self, input_ids: torch.Tensor,
                record_queries: bool = False) -> AttentionRecorder:
        """Forward pass that records per-layer post-RoPE K (and optionally Q) and V."""
        global _ACTIVE_RECORDER
        rec = AttentionRecorder()
        rec.record_queries = record_queries
        _ACTIVE_RECORDER = rec
        try:
            self.model(input_ids=input_ids.to(self.device), use_cache=False)
        finally:
            _ACTIVE_RECORDER = None
        if len(rec.keys) != self.num_layers:
            raise RuntimeError(
                f"Recorded {len(rec.keys)}/{self.num_layers} layers on {self.name}; "
                "the attention recorder was not invoked by every layer."
            )
        return rec


# --------------------------------------------------------------------------
# RoPE strip / apply (HF rotate_half convention)
# --------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [batch, heads, seq, dim]; cos/sin: [batch, seq, dim]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


def strip_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Inverse rotation: exact because RoPE is orthogonal."""
    return apply_rope(x, cos, -sin)
