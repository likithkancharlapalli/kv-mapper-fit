"""Stage 2: linear-structure analysis and source-layer selection.

For every (source layer, target layer, head) triple we fit a single-source
OLS probe in content space (RoPE-stripped keys) and value space, and record
the head-averaged R^2 (paper section 2.3, eq. 2). The per-target-layer top-k
source layers, ranked by head-averaged R^2 averaged over K_stripped and V,
drive the production ridge fit (section 3.2).

Outputs analysis.json:
  r2_k, r2_v          [L_s][L_t] head-averaged R^2 matrices
  selected_layers     [L_t][k]   source layers per target layer
and, if matplotlib is available, r2_heatmaps.png.
"""

from __future__ import annotations

import json
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from kv_mapper_fit.capture import CaptureStore
from kv_mapper_fit.config import PipelineConfig, analysis_path, heatmap_path

_JITTER = 1e-4  # relative ridge jitter for probe conditioning


def _gram(x: torch.Tensor) -> torch.Tensor:
    """x: [N, H, d] centered -> per-head Gram [H, d, d]."""
    return torch.einsum("nhd,nhe->hde", x, x)


def _probe_r2(xs: torch.Tensor, yt: torch.Tensor, gram_chol: torch.Tensor,
              ss_tot: torch.Tensor) -> float:
    """Head-averaged in-sample R^2 of per-head OLS xs -> yt (both centered).

    gram_chol: Cholesky of xs' per-head Gram (+jitter), [H, d, d].
    ss_tot: per-head total sum of squares of yt, [H].
    """
    c = torch.einsum("nhd,nhe->hde", xs, yt)          # [H, d, d]
    sol = torch.cholesky_solve(c, gram_chol)           # G^-1 C
    explained = (c * sol).sum(dim=(1, 2))              # tr(C^T G^-1 C) per head
    ss_res = ss_tot - explained
    r2 = 1.0 - ss_res / ss_tot.clamp_min(1e-12)
    return float(r2.mean())


def run_analysis(cfg: PipelineConfig) -> dict:
    workdir = cfg.resolved_workdir()
    src = CaptureStore(workdir, "source")
    tgt = CaptureStore(workdir, "target")

    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.probe_tokens, src.num_tokens)
    token_idx = np.sort(rng.choice(src.num_tokens, size=n, replace=False))

    results = {}
    for kind in ("k", "v"):
        # Preload the probe subsample of every source layer (fp16 in RAM),
        # with centered fp32 views materialized per use.
        src_layers, chols, ss_by_layer = [], [], []
        for ls in range(src.num_layers):
            x = src.layer(kind, ls, token_idx)
            x = x - x.mean(dim=0, keepdim=True)
            g = _gram(x)
            eye = torch.eye(src.head_dim).expand_as(g)
            jitter = _JITTER * g.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-8)
            chols.append(torch.linalg.cholesky(g + jitter[:, None, None] * eye))
            src_layers.append(x.half())
        r2 = np.zeros((src.num_layers, tgt.num_layers), dtype=np.float64)
        for lt in tqdm(range(tgt.num_layers), desc=f"probe[{kind}]"):
            y = tgt.layer(kind, lt, token_idx)
            y = y - y.mean(dim=0, keepdim=True)
            ss_tot = (y * y).sum(dim=(0, 2))  # [H]
            for ls in range(src.num_layers):
                r2[ls, lt] = _probe_r2(src_layers[ls].float(), y,
                                       chols[ls], ss_tot)
        results[kind] = r2

    combined = (results["k"] + results["v"]) / 2.0
    k = src.num_layers if cfg.top_k <= 0 else min(cfg.top_k, src.num_layers)
    selected: List[List[int]] = []
    for lt in range(tgt.num_layers):
        order = np.argsort(-combined[:, lt])[:k]
        selected.append(sorted(int(i) for i in order))

    payload = {
        "probe_tokens": int(n),
        "top_k": k,
        "r2_k": results["k"].round(4).tolist(),
        "r2_v": results["v"].round(4).tolist(),
        "selected_layers": selected,
        "best_r2_k": float(results["k"].max()),
        "best_r2_v": float(results["v"].max()),
        "mean_best_r2_k": float(results["k"].max(axis=0).mean()),
        "mean_best_r2_v": float(results["v"].max(axis=0).mean()),
    }
    analysis_path(workdir).write_text(json.dumps(payload, indent=2))
    _maybe_plot(workdir, results)
    print(f"analysis done: best single-source R^2  K={payload['best_r2_k']:.3f} "
          f"V={payload['best_r2_v']:.3f}; selected top-{k} layers per target layer")
    return payload


def _maybe_plot(workdir, results) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, kind, title in zip(axes, ("k", "v"),
                               ("K (RoPE-stripped)", "V")):
        im = ax.imshow(results[kind], aspect="auto", origin="lower",
                       vmin=0, vmax=1, cmap="viridis")
        ax.set_xlabel("target layer")
        ax.set_ylabel("source layer")
        ax.set_title(f"head-averaged $R^2$: {title}")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(heatmap_path(workdir), dpi=150)
    plt.close(fig)
