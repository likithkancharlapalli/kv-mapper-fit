"""Runtime application of a fitted mapper to a source KV cache.

Inference is one batched matmul per target layer (paper section 4.7): the
selected source layers' RoPE-stripped keys (or values) are concatenated on
the feature axis and projected with the layer's ridge weights; mapped keys
are then re-rotated with the *target's* RoPE at the original positions.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import torch

from kv_mapper_fit.modeling import apply_rope, strip_rope


class KVCacheMapper:
    def __init__(self, tensors: Dict[str, torch.Tensor], metadata: dict):
        self.tensors = tensors
        self.meta = metadata
        self.selected: List[List[int]] = metadata["selected_layers"]
        self.num_target_layers: int = metadata["num_target_layers"]
        self.num_kv_heads: int = metadata["num_kv_heads"]
        self.head_dim: int = metadata["head_dim"]

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "KVCacheMapper":
        from safetensors.torch import safe_open

        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device=device) as f:
            metadata = json.loads(f.metadata()["kvmf"])
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return cls(tensors, metadata)

    def to(self, device) -> "KVCacheMapper":
        self.tensors = {k: v.to(device) for k, v in self.tensors.items()}
        return self

    # ------------------------------------------------------------------
    def map_stripped(
        self,
        src_k_stripped: List[torch.Tensor],
        src_v: List[torch.Tensor],
        tgt_cos: torch.Tensor,
        tgt_sin: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Map a full source cache (content-space keys) to the target format.

        src_k_stripped / src_v: per source layer, [B, n_kv, T, d_h].
        tgt_cos / tgt_sin: target RoPE tables at the cache positions, [B, T, d_h].
        Returns per-target-layer lists in the same [B, n_kv, T, d_h] layout.
        """
        h, d = self.num_kv_heads, self.head_dim
        out_k, out_v = [], []
        for lt in range(self.num_target_layers):
            sel = self.selected[lt]
            xk = self._design(src_k_stripped, sel)          # [B, T, k*h*d]
            xv = self._design(src_v, sel)
            k_hat = xk @ self.tensors[f"layers.{lt}.k.weight"] \
                + self.tensors[f"layers.{lt}.k.bias"]
            v_hat = xv @ self.tensors[f"layers.{lt}.v.weight"] \
                + self.tensors[f"layers.{lt}.v.bias"]
            b, t = k_hat.shape[0], k_hat.shape[1]
            k_hat = k_hat.reshape(b, t, h, d).permute(0, 2, 1, 3)
            v_hat = v_hat.reshape(b, t, h, d).permute(0, 2, 1, 3)
            out_k.append(apply_rope(k_hat, tgt_cos, tgt_sin))
            out_v.append(v_hat)
        return out_k, out_v

    def map_rotated(
        self,
        src_k: List[torch.Tensor],
        src_v: List[torch.Tensor],
        src_cos: torch.Tensor,
        src_sin: torch.Tensor,
        tgt_cos: torch.Tensor,
        tgt_sin: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Same, but accepts as-stored (post-RoPE) source keys and strips first."""
        stripped = [strip_rope(k, src_cos, src_sin) for k in src_k]
        return self.map_stripped(stripped, src_v, tgt_cos, tgt_sin)

    @staticmethod
    def _design(layers: List[torch.Tensor], sel: List[int]) -> torch.Tensor:
        parts = []
        for ls in sel:
            x = layers[ls]                                   # [B, h, T, d]
            b, h, t, d = x.shape
            parts.append(x.permute(0, 2, 1, 3).reshape(b, t, h * d))
        return torch.cat(parts, dim=-1)
