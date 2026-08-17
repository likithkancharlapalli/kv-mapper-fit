"""Stage 3: closed-form per-head ridge fit and safetensors export.

For each target layer l we concatenate the KV features of its selected
source layers into a design matrix X in R^{N x (k * n_kv * d_h)} and solve

    W* = (X^T X + lambda I)^-1 X^T Y        (paper eq. 4)

with X, Y centered; the bias is recovered as b = mean(Y) - mean(X) W*.
All target heads of a layer share X, so we solve one multi-RHS system per
(layer, K|V) with Y stacking every head -- per-head independence is
preserved because each output column has its own regression.

Moments are accumulated in a single chunked pass over the capture memmaps
(float32 matmuls on the compute device, float64 accumulators), and each
solve runs in float64, matching the paper's precision regime.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from kv_mapper_fit.capture import CaptureStore
from kv_mapper_fit.config import (
    PipelineConfig, analysis_path, mapper_path,
)
from kv_mapper_fit.modeling import resolve_device

_CHUNK_ROWS = 16384


def _layer_moments(src: CaptureStore, tgt: CaptureStore, kind: str,
                   selected: List[int], lt: int, device: torch.device):
    """One pass over the data: uncentered X^T X, X^T Y, and column sums."""
    n = src.num_tokens
    d_in = len(selected) * src.num_kv_heads * src.head_dim
    d_out = tgt.num_kv_heads * tgt.head_dim
    g = torch.zeros(d_in, d_in, dtype=torch.float64)
    c = torch.zeros(d_in, d_out, dtype=torch.float64)
    yy = torch.zeros((), dtype=torch.float64)  # tr(Y^T Y), scalar suffices
    sx = torch.zeros(d_in, dtype=torch.float64)
    sy = torch.zeros(d_out, dtype=torch.float64)

    src_arr, tgt_arr = (src.k, tgt.k) if kind == "k" else (src.v, tgt.v)
    for start in range(0, n, _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, n)
        rows = stop - start
        x_parts = [
            torch.from_numpy(np.array(src_arr[ls, start:stop]))
            .float().reshape(rows, -1)
            for ls in selected
        ]
        x = torch.cat(x_parts, dim=1).to(device)
        y = torch.from_numpy(np.array(tgt_arr[lt, start:stop])) \
            .float().reshape(rows, -1).to(device)
        g += (x.T @ x).double().cpu()
        c += (x.T @ y).double().cpu()
        yy += (y * y).sum().double().cpu()
        sx += x.sum(dim=0).double().cpu()
        sy += y.sum(dim=0).double().cpu()
    return g, c, yy, sx, sy, n


def _solve_ridge(g, c, yy, sx, sy, n, lam):
    """Centered ridge solve from uncentered moments. Returns W, b, R^2."""
    mx, my = sx / n, sy / n
    g_c = g - n * torch.outer(mx, mx)
    c_c = c - n * torch.outer(mx, my)
    ss_tot = yy - n * (my * my).sum()
    d_in = g.shape[0]
    a = g_c + lam * torch.eye(d_in, dtype=torch.float64)
    chol = torch.linalg.cholesky(a)
    w = torch.cholesky_solve(c_c, chol)
    b = my - mx @ w
    # in-sample R^2 from moments: SS_res = tr(YcTYc) - 2 tr(WT Cc) + tr(WT Gc W)
    ss_res = ss_tot - 2 * (w * c_c).sum() + (w * (g_c @ w)).sum()
    r2 = float(1.0 - ss_res / ss_tot.clamp_min(1e-12))
    return w.float(), b.float(), r2


def run_fit(cfg: PipelineConfig) -> dict:
    workdir = cfg.resolved_workdir()
    src = CaptureStore(workdir, "source")
    tgt = CaptureStore(workdir, "target")
    analysis = json.loads(analysis_path(workdir).read_text())
    selected: List[List[int]] = analysis["selected_layers"]
    device = resolve_device(cfg.device)
    if device.type == "mps":
        device = torch.device("cpu")  # fp accumulation matmuls: keep it simple

    tensors: Dict[str, torch.Tensor] = {}
    r2_report = {"k": [], "v": []}
    for lt in tqdm(range(tgt.num_layers), desc="ridge fit"):
        for kind in ("k", "v"):
            moments = _layer_moments(src, tgt, kind, selected[lt], lt, device)
            w, b, r2 = _solve_ridge(*moments, lam=cfg.ridge_lambda)
            tensors[f"layers.{lt}.{kind}.weight"] = w.contiguous()
            tensors[f"layers.{lt}.{kind}.bias"] = b.contiguous()
            r2_report[kind].append(round(r2, 4))

    total_params = sum(t.numel() for t in tensors.values())
    metadata = {
        "format_version": "1",
        "source_model": cfg.source_model,
        "target_model": cfg.target_model,
        "num_source_layers": src.num_layers,
        "num_target_layers": tgt.num_layers,
        "num_kv_heads": src.num_kv_heads,
        "head_dim": src.head_dim,
        "top_k": analysis["top_k"],
        "selected_layers": selected,
        "ridge_lambda": cfg.ridge_lambda,
        "calibration_tokens": src.num_tokens,
        "r2_k_per_layer": r2_report["k"],
        "r2_v_per_layer": r2_report["v"],
        "total_params": total_params,
    }
    from safetensors.torch import save_file
    out = mapper_path(workdir)
    save_file(tensors, str(out), metadata={"kvmf": json.dumps(metadata)})
    mean_k = float(np.mean(r2_report["k"]))
    mean_v = float(np.mean(r2_report["v"]))
    print(f"fit done: layer-mean ridge R^2  K={mean_k:.3f} V={mean_v:.3f}; "
          f"{total_params/1e6:.1f}M mapper params -> {out}")
    return metadata
