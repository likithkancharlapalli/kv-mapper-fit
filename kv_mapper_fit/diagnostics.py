"""Stage 4: deployment diagnostics on held-out sequences.

Calibration R^2 does not predict downstream retention across pairs; the
attention-output cosine does (Pearson r = +0.57 vs -0.20 over the paper's 12
pair evaluations, section 4.5). We therefore compute, per layer and head, the
cosine between the target's attention output computed from mapped KV and from
its own ground-truth KV, on sequences held out from calibration, and report:

  mean_attention_cosine   the headline predictor
  eval_r2_k / eval_r2_v   held-out-domain reconstruction R^2 (deeply negative
                          eval R^2_K is the paper's failure signature)
  tier verdict            heuristic Tier 1 / borderline / Tier 2 call

Thresholds are heuristics derived from the paper's reported tiers, not
guarantees; validate on your own downstream task before shipping.
"""

from __future__ import annotations

import gc
import json
import math
from typing import List

import torch
from tqdm import tqdm

from kv_mapper_fit.config import PipelineConfig, diagnostics_path, mapper_path
from kv_mapper_fit.data import build_calibration_batches
from kv_mapper_fit.mapper import KVCacheMapper
from kv_mapper_fit.modeling import (
    CapturableModel, resolve_device, resolve_dtype, strip_rope,
)

TIER1_COSINE = 0.80
BORDERLINE_COSINE = 0.65


def _attention_output(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      scaling: float) -> torch.Tensor:
    """Causal GQA attention. q: [1, n_q, T, d]; k, v: [1, n_kv, T, d]."""
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    t = q.shape[2]
    scores = (q @ k.transpose(-1, -2)) * scaling
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _r2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    truth_c = truth - truth.mean()
    return float(1.0 - ((pred - truth) ** 2).sum() / (truth_c ** 2).sum())


def run_diagnostics(cfg: PipelineConfig) -> dict:
    workdir = cfg.resolved_workdir()
    mapper = KVCacheMapper.load(str(mapper_path(workdir)))
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.source_model, trust_remote_code=cfg.trust_remote_code
    )
    # Held out: drawn from beyond the calibration range of the same corpus.
    batches = build_calibration_batches(
        cfg, tokenizer, cfg.diag_sequences, cfg.diag_seq_len,
        skip_sequences=cfg.num_sequences,
    )
    sequences = [row.unsqueeze(0) for b in batches for row in b]

    # Pass 1: source caches (content-space keys), fp16 on CPU.
    src = CapturableModel(cfg.source_model, device, dtype, cfg.trust_remote_code)
    src_caches = []
    for seq in tqdm(sequences, desc="diagnose[source]"):
        rec = src.capture(seq)
        pos = torch.arange(seq.shape[1]).unsqueeze(0)
        cos, sin = src.rope_cos_sin(pos)
        ks = [strip_rope(rec.keys[l], cos, sin).half() for l in range(src.num_layers)]
        vs = [rec.values[l].half() for l in range(src.num_layers)]
        src_caches.append((ks, vs))
    del src
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pass 2: target ground truth (with queries), map, and compare.
    tgt = CapturableModel(cfg.target_model, device, dtype, cfg.trust_remote_code)
    scalings = _per_layer_scaling(tgt)
    cos_by_layer = torch.zeros(tgt.num_layers, dtype=torch.float64)
    r2_k_num = r2_k_den = r2_v_num = r2_v_den = 0.0
    for idx, seq in enumerate(tqdm(sequences, desc="diagnose[target]")):
        rec = tgt.capture(seq, record_queries=True)
        pos = torch.arange(seq.shape[1]).unsqueeze(0)
        tcos, tsin = tgt.rope_cos_sin(pos)
        ks, vs = src_caches[idx]
        k_hat, v_hat = mapper.map_stripped(
            [k.float() for k in ks], [v.float() for v in vs], tcos, tsin)
        for layer in range(tgt.num_layers):
            q = rec.queries[layer]
            out_gt = _attention_output(q, rec.keys[layer], rec.values[layer],
                                       scalings[layer])
            out_map = _attention_output(q, k_hat[layer], v_hat[layer],
                                        scalings[layer])
            cos_sim = torch.nn.functional.cosine_similarity(
                out_map, out_gt, dim=-1)          # [1, n_q, T]
            cos_by_layer[layer] += float(cos_sim.mean())
            k_gt_stripped = strip_rope(rec.keys[layer], tcos, tsin)
            k_hat_stripped = strip_rope(k_hat[layer], tcos, tsin)
            r2_k_num += float(((k_hat_stripped - k_gt_stripped) ** 2).sum())
            r2_k_den += float(((k_gt_stripped - k_gt_stripped.mean()) ** 2).sum())
            r2_v_num += float(((v_hat[layer] - rec.values[layer]) ** 2).sum())
            r2_v_den += float(
                ((rec.values[layer] - rec.values[layer].mean()) ** 2).sum())
        rec.clear()
    del tgt
    gc.collect()

    cos_by_layer /= len(sequences)
    mean_cosine = float(cos_by_layer.mean())
    eval_r2_k = 1.0 - r2_k_num / max(r2_k_den, 1e-12)
    eval_r2_v = 1.0 - r2_v_num / max(r2_v_den, 1e-12)

    if mean_cosine >= TIER1_COSINE and eval_r2_k > 0:
        tier, advice = "tier1", ("Linear ridge mapper looks deployable. "
                                 "Validate on your downstream task.")
    elif mean_cosine >= BORDERLINE_COSINE:
        tier, advice = "borderline", (
            "Transfer may work but expect degradation on knowledge-heavy or "
            "generation tasks. Benchmark before shipping; consider the MLP "
            "mapper variant.")
    else:
        tier, advice = "tier2", (
            "Linear mapping is likely insufficient for this pair (paper "
            "section 4.4). Fit the nonlinear MLP mapper or choose a "
            "different pair.")

    payload = {
        "mean_attention_cosine": round(mean_cosine, 4),
        "attention_cosine_per_layer": [round(float(x), 4) for x in cos_by_layer],
        "eval_r2_k": round(eval_r2_k, 4),
        "eval_r2_v": round(eval_r2_v, 4),
        "diag_sequences": len(sequences),
        "tier": tier,
        "advice": advice,
        "thresholds": {"tier1": TIER1_COSINE, "borderline": BORDERLINE_COSINE},
    }
    diagnostics_path(workdir).write_text(json.dumps(payload, indent=2))
    print(f"diagnostics done: attention-output cosine={mean_cosine:.3f}, "
          f"eval R^2 K={eval_r2_k:.3f} V={eval_r2_v:.3f} -> {tier.upper()}")
    print(f"  {advice}")
    return payload


def _per_layer_scaling(model: CapturableModel) -> List[float]:
    base = getattr(model.model, "model", model.model)
    default = model.head_dim ** -0.5
    return [
        float(getattr(layer.self_attn, "scaling", default))
        for layer in base.layers
    ]
