"""Stage 1: calibration capture.

Runs the source and target models (sequentially, to bound memory) over the
same tokenized calibration sequences and stores stride-subsampled,
RoPE-stripped keys and raw values per layer as float16 memmaps:

  capture_<role>/meta.json
  capture_<role>/k_stripped.npy   [L, N, n_kv, d_h]
  capture_<role>/v.npy            [L, N, n_kv, d_h]

Keys are stored in content space (RoPE removed) so downstream ridge fits are
position-free and reusable across context lengths (paper section 3.3).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from kv_mapper_fit.config import PipelineConfig, capture_dir
from kv_mapper_fit.data import build_calibration_batches
from kv_mapper_fit.modeling import (
    CapturableModel, resolve_device, resolve_dtype, strip_rope,
)


def _subsample_positions(seq_len: int, stride: int) -> torch.Tensor:
    return torch.arange(0, seq_len, stride, dtype=torch.long)


def _capture_one_model(cfg: PipelineConfig, model_name: str, role: str,
                       batches: List[torch.Tensor]) -> dict:
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    model = CapturableModel(model_name, device, dtype, cfg.trust_remote_code)

    keep = _subsample_positions(cfg.seq_len, cfg.subsample_stride)
    tokens_per_seq = keep.numel()
    total_tokens = sum(b.shape[0] for b in batches) * tokens_per_seq

    out = capture_dir(cfg.resolved_workdir(), role)
    out.mkdir(parents=True, exist_ok=True)
    shape = (model.num_layers, total_tokens, model.num_kv_heads, model.head_dim)
    k_mm = np.lib.format.open_memmap(out / "k_stripped.npy", mode="w+",
                                     dtype=np.float16, shape=shape)
    v_mm = np.lib.format.open_memmap(out / "v.npy", mode="w+",
                                     dtype=np.float16, shape=shape)

    cursor = 0
    for batch in tqdm(batches, desc=f"capture[{role}] {model_name}"):
        bsz = batch.shape[0]
        rec = model.capture(batch)
        positions = torch.arange(cfg.seq_len).unsqueeze(0).expand(bsz, -1)
        cos, sin = model.rope_cos_sin(positions)
        n = bsz * tokens_per_seq
        for layer in range(model.num_layers):
            k = strip_rope(rec.keys[layer], cos, sin)      # [b, h, T, d]
            v = rec.values[layer]
            # subsample tokens, then flatten batch*tokens -> N axis
            k = k[:, :, keep, :].permute(0, 2, 1, 3).reshape(n, model.num_kv_heads,
                                                             model.head_dim)
            v = v[:, :, keep, :].permute(0, 2, 1, 3).reshape(n, model.num_kv_heads,
                                                             model.head_dim)
            k_mm[layer, cursor:cursor + n] = k.numpy().astype(np.float16)
            v_mm[layer, cursor:cursor + n] = v.numpy().astype(np.float16)
        cursor += n
        rec.clear()

    k_mm.flush()
    v_mm.flush()
    meta = {
        "model": model_name,
        "num_layers": model.num_layers,
        "num_kv_heads": model.num_kv_heads,
        "head_dim": model.head_dim,
        "num_tokens": total_tokens,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def run_capture(cfg: PipelineConfig) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.source_model, trust_remote_code=cfg.trust_remote_code
    )
    tgt_tok = AutoTokenizer.from_pretrained(
        cfg.target_model, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.vocab_size != tgt_tok.vocab_size:
        raise RuntimeError(
            "Source and target tokenizers differ (vocab "
            f"{tokenizer.vocab_size} vs {tgt_tok.vocab_size}). Cross-model KV "
            "transfer requires a shared tokenizer within the family."
        )

    batches = build_calibration_batches(cfg, tokenizer, cfg.num_sequences,
                                        cfg.seq_len)
    src_meta = _capture_one_model(cfg, cfg.source_model, "source", batches)
    tgt_meta = _capture_one_model(cfg, cfg.target_model, "target", batches)

    if (src_meta["num_kv_heads"] != tgt_meta["num_kv_heads"]
            or src_meta["head_dim"] != tgt_meta["head_dim"]):
        raise RuntimeError(
            "Pair is not matched-KV: source has "
            f"{src_meta['num_kv_heads']}x{src_meta['head_dim']}, target has "
            f"{tgt_meta['num_kv_heads']}x{tgt_meta['head_dim']}. The paper's "
            "closed-form recipe is validated only on matched-KV pairs."
        )
    cfg.save()
    print(f"capture done: {src_meta['num_tokens']} calibration tokens, "
          f"source L={src_meta['num_layers']}, target L={tgt_meta['num_layers']}")


class CaptureStore:
    """Read-side view of a capture directory (memmapped, lazy)."""

    def __init__(self, workdir: Path, role: str):
        d = capture_dir(workdir, role)
        self.meta = json.loads((d / "meta.json").read_text())
        self.k = np.load(d / "k_stripped.npy", mmap_mode="r")
        self.v = np.load(d / "v.npy", mmap_mode="r")

    @property
    def num_layers(self) -> int:
        return self.meta["num_layers"]

    @property
    def num_kv_heads(self) -> int:
        return self.meta["num_kv_heads"]

    @property
    def head_dim(self) -> int:
        return self.meta["head_dim"]

    @property
    def num_tokens(self) -> int:
        return self.meta["num_tokens"]

    def layer(self, kind: str, layer: int, token_idx=None) -> torch.Tensor:
        """Load one layer as float32 [N, n_kv, d_h]; optionally row-subsampled."""
        arr = self.k if kind == "k" else self.v
        data = arr[layer] if token_idx is None else arr[layer][token_idx]
        return torch.from_numpy(np.array(data)).float()
