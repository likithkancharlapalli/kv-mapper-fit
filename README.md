# kv-mapper-fit

**A calibration + fitting toolkit for cross-model KV cache transfer within LLM
families.** Point it at two HuggingFace models from the same family (e.g.
Qwen3 14B and Qwen3 32B) and it produces a *mapper* — a set of per-layer
linear projections that convert one model's prefilled KV cache into the
other's expected format, so the receiving model can start decoding
immediately **without re-running prefill**.

Implements the closed-form recipe from
[*Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping
for Prefill Reuse*](https://arxiv.org/abs/2608.03893) (NVIDIA, 2026).

---

## The problem this solves

Production LLM serving increasingly swaps between different-sized models in
the same family:

- **Cost-quality cascading** — answer easy queries with the small model,
  escalate hard ones to the large model.
- **Mid-conversation switching** — a router upgrades or downgrades the model
  partway through a long session.
- **Speculative decoding** — a small draft model works alongside a large
  target model on the same prompt.

Every one of these swaps has a hidden tax: the receiving model must re-run
**prefill** — the forward pass that populates the KV cache — over the entire
accumulated context. For long agentic sessions this is often the dominant
latency cost, and prefix caching doesn't help because it only works within a
single model.

The paper's key finding is that the relationship between two family members'
KV caches is **largely linear**, so a simple ridge regression — fit in closed
form from a small calibration set, no gradient training — can translate one
model's cache into the other's format. On the good pairs this retains
73–98% of the receiver's standalone accuracy while running **2.7–25× faster
than re-prefill**.

```
  ┌──────────────┐  prefill   ┌────────────┐   linear    ┌────────────┐  decode
  │ prompt/hist. │ ─────────> │ 14B's KV   │ ──────────> │ 32B-format │ ───────>
  └──────────────┘  (cheap,   │ cache      │  mapper     │ KV cache   │  on 32B,
                     on 14B)  └────────────┘  (matmuls)  └────────────┘  no prefill
```

## How the method works

The mapper is fit once per (source → target) direction and reused for every
request. Three ideas from the paper make it work:

**1. Per-head closed-form ridge regression (§3.1).**
For each target layer and KV head, the mapping from source KV features to
target keys (or values) is solved exactly:

```
W* = (XᵀX + λI)⁻¹ XᵀY        λ = 0.01
```

`X` stacks ~128K calibration tokens' source features, `Y` the target model's
own keys/values for the same tokens. No backprop, no epochs — one linear
solve per (layer, K|V), with the Gram matrix shared across a layer's heads.

**2. Cross-layer source selection (§3.2).**
Source and target have different depths, and no single source layer carries
all the information a target layer needs. For each target layer, the toolkit
probes every source layer with a cheap single-source OLS fit, ranks them by
head-averaged R², and concatenates the **top-k** most predictive ones as the
regression input. This is the single largest contributor to quality in the
paper's ablations.

**3. Content-space (RoPE-stripped) mapping (§3.3).**
Keys in the cache carry a position-dependent rotary encoding (RoPE). The
toolkit removes it before fitting — the rotation is orthogonal, so this is
exact — fits the mapping in position-free "content space", and re-applies the
*target's* RoPE after mapping. The result: one mapper works at any context
length, not just the 1,024-token length seen during calibration.

**Knowing when it will work (§4.5).**
The paper's most useful diagnostic finding: calibration R² does *not*
predict downstream retention across pairs (r = −0.20), but the
**attention-output cosine** — how similar the target's attention outputs are
when computed from mapped vs. ground-truth KV — does (r = +0.57). The toolkit
computes this on held-out data and issues a Tier 1 / borderline / Tier 2
verdict, so you know whether a pair is deployable *before* running downstream
benchmarks. Tier 2 pairs need the paper's nonlinear MLP variant (roadmap).

## The pipeline

Four stages, each a CLI subcommand, all sharing a work directory:

| Stage | What it does | Output |
|---|---|---|
| `capture` | Streams 500 FineWeb-Edu sequences (1,024 tokens) through both models; records per-layer RoPE-stripped keys + values, stride-4 subsampled | `capture_{source,target}/` fp16 memmaps |
| `analyze` | Single-source R² probe for every (source layer, target layer) pair; top-k selection per target layer | `analysis.json`, `r2_heatmaps.png` |
| `fit` | Closed-form ridge solve per (target layer, K\|V), float64 solve, in-sample R² report | `mapper.safetensors` |
| `diagnose` | Attention-output cosine + eval-domain R² on held-out sequences; tier verdict | `diagnostics.json` |

Defaults match the paper's production recipe (Appendix E). `capture` refuses
pairs that aren't matched-KV (same KV head count and head dim) or don't share
a tokenizer, since the recipe is only validated under those conditions.

## Install

```bash
pip install -e ".[data,plots]"
```

- `data` → `datasets`, for streaming FineWeb-Edu (no full download)
- `plots` → `matplotlib`, for the R² heatmap PNG

Both are optional: pass `--text-file corpus.txt` to use a local
newline-delimited corpus fully offline.

## Usage

Full pipeline with paper defaults:

```bash
kv-mapper-fit run \
  --source Qwen/Qwen3-14B \
  --target Qwen/Qwen3-32B \
  -w work/qwen3-14b-to-32b
```

Or stage by stage:

```bash
kv-mapper-fit capture  --source Qwen/Qwen3-14B --target Qwen/Qwen3-32B -w work/pair
kv-mapper-fit analyze  -w work/pair --top-k 8          # or --top-k all
kv-mapper-fit fit      -w work/pair --ridge-lambda 0.01
kv-mapper-fit diagnose -w work/pair
```

Useful knobs: `--num-sequences` / `--seq-len` / `--subsample-stride` (capture
cost), `--probe-tokens` (analysis cost), `--device` / `--dtype` /
`--batch-size` (runtime).

### Applying a mapper at inference

```python
from kv_mapper_fit import KVCacheMapper

mapper = KVCacheMapper.load("work/pair/mapper.safetensors", device="cuda")

# src_k_stripped / src_v: per source layer, [B, n_kv, T, d_h]
# tgt_cos / tgt_sin: the *target's* RoPE tables at the cache positions
tgt_k, tgt_v = mapper.map_stripped(src_k_stripped, src_v, tgt_cos, tgt_sin)
# feed tgt_k / tgt_v into the target's past_key_values and decode
```

`mapper.map_rotated(...)` accepts as-stored (post-RoPE) source keys and
strips them first. Inference cost is one batched matmul per target layer —
it does not grow with model size, only with `k`, target depth, and sequence
length.

### Reading the diagnosis

| Signal | Verdict |
|---|---|
| cosine ≥ 0.80 and eval R²_K > 0 | **Tier 1** — linear mapper likely retains most accuracy |
| cosine 0.65–0.80 | **Borderline** — benchmark before shipping |
| cosine < 0.65, or eval R²_K deeply negative | **Tier 2** — linear map insufficient; the paper's MLP variant recovered up to +37 pp HellaSwag retention on such pairs |

Thresholds are heuristics calibrated to the paper's reported tiers, not
guarantees. Always validate on your downstream task.

## Code layout

```
kv_mapper_fit/
├── cli.py          argparse subcommands (capture / analyze / fit / diagnose / run)
├── config.py       PipelineConfig + workdir artifact paths
├── data.py         FineWeb-Edu streaming or local corpus → fixed-length batches
├── modeling.py     model loading, attention recorder (captures post-RoPE Q/K
│                   and V via transformers' attention-function registry, so
│                   per-head QK-norm like Qwen3's is handled correctly),
│                   RoPE strip/apply
├── capture.py      stage 1: calibration KV extraction → fp16 memmaps
├── analysis.py     stage 2: R² probes (Cholesky-based, per-head), heatmaps,
│                   top-k selection
├── ridge.py        stage 3: chunked single-pass moment accumulation,
│                   float64 centered ridge solve, safetensors export
├── diagnostics.py  stage 4: attention-output cosine, eval R², tier verdict
└── mapper.py       KVCacheMapper — runtime application of a fitted mapper
tests/
└── test_pipeline.py  offline end-to-end tests on tiny random models
scripts/
└── make_tiny_assets.py  generate the tiny models/corpus for local demos
```

## Verifying the pipeline offline

The test suite exercises the whole pipeline on tiny random Llama-style models
with no downloads or GPU. The strongest check is **self-transfer**: mapping a
model to itself is exactly linear, so probe R², ridge R², eval R², and
attention cosine must all come out ≈ 1.0 — which validates the RoPE
round-trip, centering, solve, export/load, and attention recomputation in one
shot.

```bash
pip install -e ".[dev]" tokenizers
pytest tests/ -q
```

## Requirements and scope

- **Matched-KV pairs only** — source and target must share KV head count and
  per-head dim (true across scales for Qwen3, Llama 3.1, Ministral 3, and
  most modern families). Checked at capture time.
- **Shared tokenizer** within the family (checked).
- **Dense full-attention decoders with per-layer RoPE** (Llama/Qwen/Mistral
  style). Sliding-window and attention-recurrent hybrids are out of scope,
  matching the paper.
- **Directional** — fit one mapper per (source → target) direction.
- **Compute** — the paper fits on one 8×H100 node in ~1 hour per pair.
  Calibration scales down gracefully (N=50 sequences stays within ~1.6 pp in
  the paper's ablation), so small pairs fit on much less. Capture storage is
  `2 · L · N_tokens · n_kv · d_h` fp16 per model (~34 GB for a Qwen3-32B
  target at paper defaults; shrink with `--num-sequences` /
  `--subsample-stride`).
- **Matched-KV is necessary, not sufficient** — two of the paper's six pairs
  degraded badly under the linear map. That's exactly what the `diagnose`
  stage screens for.

## Roadmap

- [ ] MLP mapper variant (paper §4.4) for Tier 2 pairs
- [ ] vLLM / SGLang serving hook (map-on-swap instead of re-prefill)
- [ ] K/V error-concentration diagnostics (paper Table 4)
- [ ] Multi-GPU capture sharding for 70B-class targets

## Citation

```bibtex
@article{heo2026crossmodel,
  title={Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear
         Mapping for Prefill Reuse},
  author={Heo, Taekyung and Shafipour, Rasoul and Zhao, Ritchie and others},
  journal={arXiv preprint arXiv:2608.03893},
  year={2026}
}
```
