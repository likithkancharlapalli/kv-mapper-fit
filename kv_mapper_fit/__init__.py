"""kv-mapper-fit: closed-form cross-model KV cache transfer toolkit.

Implements the calibration / analysis / fitting / diagnostics recipe from
"Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping
for Prefill Reuse" (arXiv:2608.03893).

Pipeline stages (each is a CLI subcommand, all share a work directory):

  capture   Run source + target models over a calibration corpus and dump
            RoPE-stripped keys and values per layer to disk.
  analyze   Fit single-source per-head OLS probes to build the R^2 heatmap
            and select the top-k source layers per target layer.
  fit       Solve the per-head ridge regressions in closed form and export
            the mapper as a safetensors file.
  diagnose  Compute attention-output cosine between mapped and ground-truth
            KV on held-out sequences, the paper's cross-pair retention
            predictor, and report a Tier 1 / Tier 2 verdict.
  run       All of the above, in order.
"""

__version__ = "0.1.0"

from kv_mapper_fit.mapper import KVCacheMapper  # noqa: F401
